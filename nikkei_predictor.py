#!/usr/bin/env python3
"""
日経平均株価 2営業日後 騰落予測モデル
=====================================

予測対象 : 判定日（基準日 t の翌営業日）の朝に、翌営業日（t+2）の引値が判定日（t+1）の引値より上か下か
実行時刻 : 東京市場の寄付き前（NY市場の引け後、目安 6:30〜8:30 JST）
特徴量   : 日経平均のテクニカル指標 + 日経平均と相関の高い海外マーケットデータ
モデル   : ロジスティック回帰（回帰分析ベースの分類）と勾配ブースティングを
           ウォークフォワード検証で比較し、成績の良い方で予測
運用     : 実行するたびにデータを差分更新 → 相関で市場を選び直し → 全期間で再学習

使い方:
    python nikkei_predictor.py            # 通常実行（データ更新・再学習・予測）
    python nikkei_predictor.py --demo     # ネット接続なしの動作確認（疑似データ）
    python nikkei_predictor.py --no-mail  # メール送信しない

メール送信（任意）: 環境変数 GMAIL_USER, GMAIL_APP_PASSWORD, MAIL_TO を設定
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import warnings
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 設定
JST = timezone(timedelta(hours=9))
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output"
START_DATE = "2012-01-01"

HORIZON = 2              # 何営業日後を予測するか
CORR_WINDOW = 500        # 相関を測る期間（営業日）
CORR_MIN_ABS = 0.15      # この絶対値以上の相関がある市場を採用
CORR_MIN_KEEP = 5        # 最低でも上位この数の市場は採用

TARGET = "^N225"
CANDIDATES = {           # 相関チェックの候補（Yahoo Finance のティッカー）
    "^GSPC": "S&P500",
    "^DJI": "NYダウ",
    "^IXIC": "ナスダック総合",
    "^SOX": "フィラデルフィア半導体",
    "^VIX": "VIX恐怖指数",
    "NKD=F": "日経225先物(CME)",
    "JPY=X": "ドル円",
    "EURJPY=X": "ユーロ円",
    "^TNX": "米10年金利",
    "CL=F": "WTI原油",
    "GC=F": "金",
    "^HSI": "香港ハンセン",
    "000001.SS": "上海総合",
}
NAMES = {TARGET: "日経平均", **CANDIDATES}


# ---------------------------------------------------------------- データ取得（差分更新）
def _safe(t: str) -> str:
    return t.replace("^", "").replace("=", "_").replace(".", "_")


def _download(ticker: str, start: str) -> pd.DataFrame:
    """Yahoo Finance のチャートAPIから日足を取得（yfinance不要）"""
    import time
    import requests
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
    params = {"period1": int(pd.Timestamp(start).timestamp()), "period2": int(time.time()) + 86400,
              "interval": "1d", "events": "history"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    for attempt in range(5):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 429:          # レート制限 → 待って再試行
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    off = res["meta"].get("gmtoffset", 0)
    idx = pd.to_datetime(np.array(res["timestamp"]) + off, unit="s").normalize()
    df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                       "Close": q["close"], "Volume": q["volume"]}, index=idx, dtype=float)
    time.sleep(1)
    return df.dropna(subset=["Close"])


def update_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """保存済みCSVに直近分を追記して返す（初回は全期間ダウンロード）"""
    DATA_DIR.mkdir(exist_ok=True)
    out = {}
    for t in tickers:
        path = DATA_DIR / f"{_safe(t)}.csv"
        old = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
        start = START_DATE if old is None or old.empty else (old.index.max() - timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            new = _download(t, start)
        except Exception as e:  # 1銘柄の失敗で全体を止めない
            print(f"[warn] {t} の取得に失敗: {e}")
            new = pd.DataFrame()
        df = new if old is None else pd.concat([old, new])
        if df is None or df.empty:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(path)
        out[t] = df
    return out


def demo_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """動作確認用の疑似データ（海外市場の前日リターンが日経に波及する構造）"""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(START_DATE, datetime.now(JST).date() - timedelta(days=1))
    us = rng.normal(0, 0.01, len(idx))
    out = {}
    for i, t in enumerate(tickers):
        if t == TARGET:
            r = 0.5 * np.roll(us, 1) + rng.normal(0, 0.009, len(idx))
        else:
            r = (0.8 if i < 5 else 0.1) * us + rng.normal(0, 0.008, len(idx))
        c = 100 * np.exp(np.cumsum(r))
        out[t] = pd.DataFrame({"Open": c * (1 - r / 2), "High": c * 1.005, "Low": c * 0.995,
                               "Close": c, "Volume": rng.integers(1e8, 2e8, len(idx))}, index=idx)
    return out


def drop_incomplete_today(nk: pd.DataFrame) -> pd.DataFrame:
    """東京の取引時間中に実行した場合、未確定の当日足を除外"""
    now = datetime.now(JST)
    if nk.index[-1].date() == now.date() and now.hour < 16:
        return nk.iloc[:-1]
    return nk


# ---------------------------------------------------------------- 特徴量
def align(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """他市場の系列を日経の営業日カレンダーに合わせる（休場日は直前値）"""
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def technical_features(nk: pd.DataFrame) -> pd.DataFrame:
    c, h, l = nk["Close"], nk["High"], nk["Low"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=nk.index)
    for n in (1, 2, 5, 10, 20):
        f[f"nk_ret{n}"] = np.log(c).diff(n)
    for n in (5, 25, 75):
        f[f"nk_kairi{n}"] = c / c.rolling(n).mean() - 1          # 移動平均乖離率
    f["nk_ma5_25"] = c.rolling(5).mean() / c.rolling(25).mean() - 1  # GC/DC の度合い
    # RSI(14)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["nk_rsi14"] = 100 - 100 / (1 + up / dn)
    # MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    f["nk_macd"] = macd / c
    f["nk_macd_hist"] = (macd - sig) / c
    # ボリンジャーバンド %B
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    f["nk_bb_pctb"] = (c - (m - 2 * s)) / (4 * s)
    # ストキャスティクス %K
    f["nk_stoch14"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    # ATR / ボラティリティ
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["nk_atr14"] = tr.rolling(14).mean() / c
    f["nk_vol20"] = lr.rolling(20).std()
    f["nk_vol_ratio"] = lr.rolling(5).std() / lr.rolling(60).std()
    # 出来高（データが入っている場合のみ）
    v = nk["Volume"].replace(0, np.nan)
    if v.notna().mean() > 0.8:
        f["nk_volume_z"] = np.log(v / v.rolling(20).mean())
    # --- 追加テクニカル ---
    for k in (1, 2, 3):
        f[f"nk_lag{k}"] = lr.shift(k)                               # 数日前の日次リターン
    f["nk_dow"] = nk.index.dayofweek                                 # 曜日
    f["nk_month_end"] = (nk.index.days_in_month - nk.index.day <= 3).astype(int)  # 月末接近
    upd = (lr > 0).astype(int)
    f["nk_streak"] = upd.groupby((upd != upd.shift()).cumsum()).cumcount() * np.where(upd == 1, 1, -1)  # 連騰(+)/連敗(-)
    f["nk_hi52"] = c / c.rolling(250).max() - 1                      # 52週高値からの距離
    f["nk_lo52"] = c / c.rolling(250).min() - 1                      # 52週安値からの距離
    f["nk_rsi_chg"] = f["nk_rsi14"].diff(3)                          # RSIの3日変化
    f["nk_vol_chg"] = f["nk_vol20"].pct_change(5)                    # ボラの5日変化
    return f


def market_returns(prices: dict, idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    out = {}
    for t in CANDIDATES:
        if t in prices and len(prices[t]) > CORR_WINDOW:
            c = align(prices[t]["Close"], idx)
            out[t] = c.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return out


def select_markets(nk_close: pd.Series, mret: dict[str, pd.Series]) -> tuple[pd.DataFrame, list[str]]:
    """
    直近 CORR_WINDOW 日で、各市場の当日リターンと
    ・日経の同日リターン（一般的な意味での相関）
    ・日経の翌営業日リターン（予測に効く相関）
    を計算し、翌日相関の絶対値で採用市場を決める
    """
    nk_r = nk_close.pct_change()
    rows = []
    for t, r in mret.items():
        d = pd.concat([r, nk_r, nk_r.shift(-1)], axis=1).dropna().iloc[-CORR_WINDOW:]
        rows.append({"ticker": t, "name": NAMES[t],
                     "corr_same_day": d.iloc[:, 0].corr(d.iloc[:, 1]),
                     "corr_next_day": d.iloc[:, 0].corr(d.iloc[:, 2])})
    tbl = pd.DataFrame(rows).set_index("ticker")
    tbl["abs_next"] = tbl["corr_next_day"].abs()
    tbl = tbl.sort_values("abs_next", ascending=False)
    sel = list(tbl.index[tbl["abs_next"] >= CORR_MIN_ABS])
    if len(sel) < CORR_MIN_KEEP:
        sel = list(tbl.index[:CORR_MIN_KEEP])
    tbl["selected"] = tbl.index.isin(sel)
    return tbl.drop(columns="abs_next"), sel


def market_features(prices: dict, mret: dict, sel: list[str], idx) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    for t in sel:
        k = _safe(t).lower()
        r = mret[t]
        f[f"{k}_ret1"] = r
        f[f"{k}_ret5"] = (1 + r).rolling(5).apply(np.prod, raw=True) - 1
        if t == "^VIX":
            f["vix_level"] = align(prices[t]["Close"], idx)
    if "NKD=F" in prices:   # CME日経先物と日経平均の価格差（夜間の先行き）
        f["nk_fut_basis"] = align(prices["NKD=F"]["Close"], idx) / prices[TARGET]["Close"].reindex(idx) - 1
    return f


# ---------------------------------------------------------------- 学習・検証
def models() -> dict:
    return {
        "ロジスティック回帰": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "勾配ブースティング": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                                     max_iter=300, l2_regularization=1.0,
                                                     random_state=0),
    }


def walk_forward(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """過去で学習→未来で検証 を5回。2日先ターゲットの重なりを避けるため gap を空ける"""
    tss = TimeSeriesSplit(n_splits=5, test_size=250, gap=HORIZON)
    res = []
    for name, mdl in models().items():
        for k, (tr, te) in enumerate(tss.split(X)):
            mdl.fit(X.iloc[tr], y.iloc[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            yt = y.iloc[te]
            res.append({"model": name, "fold": k, "accuracy": accuracy_score(yt, p > 0.5),
                        "auc": roc_auc_score(yt, p), "baseline": max(yt.mean(), 1 - yt.mean())})
    return pd.DataFrame(res)


def explain_logit(pipe, x_last: pd.DataFrame, top=5) -> list[tuple[str, float]]:
    scaler, lr = pipe.steps[0][1], pipe.steps[-1][1]
    contrib = lr.coef_[0] * scaler.transform(x_last)[0]
    order = np.argsort(-np.abs(contrib))[:top]
    return [(x_last.columns[i], float(contrib[i])) for i in order]


# ---------------------------------------------------------------- 予測ログ
def update_log(pred_row: dict, nk_close: pd.Series) -> pd.DataFrame:
    path = OUT_DIR / "predictions.csv"
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([pred_row])], ignore_index=True)
    log = log.drop_duplicates(subset="base_date", keep="last")
    # 結果が確定した過去予測の答え合わせ
    dates = list(nk_close.index.strftime("%Y-%m-%d"))
    actual, hit = [], []
    for _, r in log.iterrows():
        if r["base_date"] in dates and dates.index(r["base_date"]) + HORIZON < len(dates):
            i = dates.index(r["base_date"])
            # 判定日(i+1)の引値 → 翌営業日(i+2)の引値
            a = int(nk_close.iloc[i + HORIZON] > nk_close.iloc[i + HORIZON - 1])
            actual.append(a)
            hit.append(int(a == int(r["pred_up"])))
        else:
            actual.append(np.nan)
            hit.append(np.nan)
    log["actual_up"], log["hit"] = actual, hit
    log.to_csv(path, index=False)
    return log


# ---------------------------------------------------------------- 日付・表示
def is_tse_holiday(d) -> bool:
    """東証の休場日（土日・祝日・年末年始 12/31〜1/3）"""
    d = pd.Timestamp(d)
    if d.weekday() >= 5 or (d.month == 12 and d.day == 31) or (d.month == 1 and d.day <= 3):
        return True
    try:
        import jpholiday
        return bool(jpholiday.is_holiday(d.date()))
    except ImportError:
        return False


def next_business_day(d) -> pd.Timestamp:
    d = pd.Timestamp(d) + pd.Timedelta(days=1)
    while is_tse_holiday(d):
        d += pd.Timedelta(days=1)
    return d


def jp_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.month}/{d.day}({'月火水木金土日'[d.weekday()]})"


def prev_or_same_business_day(d) -> pd.Timestamp:
    d = pd.Timestamp(d)
    while is_tse_holiday(d):
        d -= pd.Timedelta(days=1)
    return d


def event_notes(target) -> list[str]:
    """予想対象日が、値動きの大きくなりやすいイベントの影響日なら説明文を返す
    - 米雇用統計: 毎月第1金曜（近似）の夜に発表 → 翌営業日
    - FOMC: 米国の決定日の翌朝（日本時間）に結果 → その日以降の最初の営業日
    - メジャーSQ: 3・6・9・12月の第2金曜（休場なら前営業日）
    FOMC の日程は data/events.csv（年1回、FRBの公表日程に合わせて追記する）
    """
    t = pd.Timestamp(target).normalize()
    notes = []
    first = pd.Timestamp(t.year, t.month, 1) - pd.Timedelta(days=7)
    for fri in pd.date_range(first, t, freq="WOM-1FRI"):
        if next_business_day(fri) == t:
            notes.append(f"米雇用統計（{jp_date(fri)}夜に発表、近似日）の結果を受ける日")
    if t.month in (3, 6, 9, 12):
        sq = pd.date_range(pd.Timestamp(t.year, t.month, 1), periods=1, freq="WOM-2FRI")[0]
        if prev_or_same_business_day(sq) == t:
            notes.append("メジャーSQ（先物・オプションの特別清算）の日")
    path = DATA_DIR / "events.csv"
    if path.exists():
        ev = pd.read_csv(path, parse_dates=["date"])
        for d in ev.loc[ev.event == "FOMC", "date"]:
            eff = d + pd.Timedelta(days=1)
            if is_tse_holiday(eff):
                eff = next_business_day(eff)
            if eff == t:
                notes.append(f"FOMC（米国 {jp_date(d)} 決定）の結果を受ける日")
        if ev.date.max() < t + pd.Timedelta(days=60):
            notes.append("（data/events.csv の日程が残り少ないため、FOMC・日銀の日程の追加が必要です）")
    return notes


_MARKET_JA = {"gspc": "S&P500", "dji": "NYダウ", "ixic": "ナスダック", "sox": "SOX半導体",
              "vix": "VIX", "nkd_f": "CME日経先物", "jpy_x": "ドル円", "eurjpy_x": "ユーロ円",
              "tnx": "米10年金利", "cl_f": "WTI原油", "gc_f": "金", "hsi": "ハンセン",
              "000001_ss": "上海総合"}
_TECH_JA = {"nk_rsi14": "日経RSI(14)", "nk_macd": "日経MACD", "nk_macd_hist": "日経MACDヒストグラム",
            "nk_bb_pctb": "日経ボリンジャー%B", "nk_stoch14": "日経ストキャス%K", "nk_atr14": "日経ATR",
            "nk_vol20": "日経20日ボラ", "nk_vol_ratio": "日経ボラ比(短期/長期)",
            "nk_ma5_25": "日経5日線/25日線", "nk_volume_z": "日経出来高(20日比)", "vix_level": "VIX水準",
            "nk_lag1": "日経1日前のリターン", "nk_lag2": "日経2日前のリターン", "nk_lag3": "日経3日前のリターン",
            "nk_dow": "曜日", "nk_month_end": "月末接近", "nk_streak": "日経連騰(+)/連敗(-)日数",
            "nk_hi52": "52週高値からの距離", "nk_lo52": "52週安値からの距離", "nk_rsi_chg": "RSIの3日変化",
            "nk_vol_chg": "ボラの5日変化", "nk_fut_basis": "CME日経先物と日経平均の差"}


def feature_ja(name: str) -> str:
    if name in _TECH_JA:
        return _TECH_JA[name]
    if name.startswith("nk_ret"):
        return f"日経{name[6:]}日リターン"
    if name.startswith("nk_kairi"):
        return f"日経{name[8:]}日線乖離率"
    for k, ja in _MARKET_JA.items():
        if name.startswith(k + "_ret"):
            n = name[len(k) + 4:]
            return f"{ja} {'前日' if n == '1' else n + '日'}リターン"
    return name


# ---------------------------------------------------------------- メール
def send_mail(subject: str, body: str) -> None:
    user, pw, to = os.getenv("GMAIL_USER"), os.getenv("GMAIL_APP_PASSWORD"), os.getenv("MAIL_TO")
    if not (user and pw and to):
        print("[info] メール設定がないため送信をスキップ")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pw)
        s.send_message(msg)
    print(f"[info] {to} に送信しました")


# ---------------------------------------------------------------- メイン
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="疑似データで動作確認")
    ap.add_argument("--no-mail", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    tickers = [TARGET, *CANDIDATES]
    prices = demo_prices(tickers) if args.demo else update_prices(tickers)
    nk = drop_incomplete_today(prices[TARGET])
    idx = nk.index

    # 1) 相関で使う市場を選ぶ（毎回選び直す）
    mret = market_returns(prices, idx)
    corr_tbl, sel = select_markets(nk["Close"], mret)

    # 2) 特徴量とターゲット
    X_all = technical_features(nk).join(market_features(prices, mret, sel, idx))
    # ターゲット: 判定日(基準日の翌営業日)の引値に対し、その翌営業日の引値が上か
    y_all = (nk["Close"].shift(-HORIZON) > nk["Close"].shift(-(HORIZON - 1))).astype(float)
    y_all[nk["Close"].shift(-HORIZON).isna()] = np.nan

    # 予測の基準日 = 採用市場がすべてその日の終値まで揃っている最新の日経営業日
    # （夕方に実行すると米国の当日終値がまだ無いため、1日前を基準日にする）
    if not args.demo:
        cutoff = min(prices[t].index.max() for t in sel)
        if cutoff < idx[-1]:
            print(f"[info] 海外市場の終値が {cutoff.date()} までのため、基準日を {cutoff.date()} 以前にします")
            nk = nk[nk.index <= cutoff]
            idx = nk.index
            X_all = X_all.loc[idx]
            y_all = y_all.loc[idx]
    x_last = X_all.ffill().iloc[[-1]]                # 予測に使う最新行
    train = X_all.join(y_all.rename("y")).dropna()
    X, y = train.drop(columns="y"), train["y"].astype(int)

    # 3) ウォークフォワード検証 → モデル選択
    wf = walk_forward(X, y)
    summary = wf.groupby("model")[["accuracy", "auc", "baseline"]].mean()
    best = summary["auc"].idxmax()

    # 4) 全期間で再学習して予測
    mdl = models()[best].fit(X, y)
    prob_up = float(mdl.predict_proba(x_last)[0, 1])
    joblib.dump({"model": mdl, "features": list(X.columns), "markets": sel}, OUT_DIR / "model.joblib")

    base_date = idx[-1].strftime("%Y-%m-%d")
    judge_day = next_business_day(idx[-1])          # 判定日（この朝に予想する）
    target_day = next_business_day(judge_day)       # 予想対象日（この日の引値を予想）
    log = update_log({"run_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M"), "base_date": base_date,
                      "judge_date": judge_day.strftime("%Y-%m-%d"),
                      "target_date": target_day.strftime("%Y-%m-%d"),
                      "base_close": round(float(nk["Close"].iloc[-1]), 2), "prob_up": round(prob_up, 4),
                      "pred_up": int(prob_up > 0.5), "model": best}, nk["Close"])

    # 5) レポート
    lr_pipe = models()["ロジスティック回帰"].fit(X, y)
    drivers = explain_logit(lr_pipe, x_last)
    done = log.dropna(subset=["hit"])
    direction = "上昇" if prob_up > 0.5 else "下落"
    conf = abs(prob_up - 0.5) * 2

    base_close = float(nk["Close"].iloc[-1])
    notes = event_notes(target_day)
    subject = (f"[日経予測]{'【値動き注意】' if notes else ''} {jp_date(target_day)}の引値は"
               f"{jp_date(judge_day)}の引値より{direction}  (上昇確率 {prob_up:.0%})")
    caution = ([f"■ 値動き注意: {jp_date(target_day)} は値動きが大きくなりやすい日です",
                *[f"  ・{n}" for n in notes],
                "  （2016年以降、こうした日の値動きは普段の1.1〜1.4倍。予想が外れたときの振れ幅も大きくなります）",
                ""] if notes else [])
    lines = [
        f"■ 日経平均 翌営業日の引値予想{'（デモ）' if args.demo else ''}",
        f"判定日    : {jp_date(judge_day)} 朝",
        f"予想対象  : {jp_date(target_day)} の引値",
        f"比較する値: {jp_date(judge_day)} の引値（当日の引けで確定）",
        f"参考      : 前日 {jp_date(idx[-1])} の引値 {base_close:,.2f} 円",
        "",
        f"予測: {direction}  （{jp_date(target_day)}の引値が{jp_date(judge_day)}の引値を上回る確率 {prob_up:.1%}"
        f" / 確信度 {conf:.0%}）",
        f"使用モデル: {best}",
        "",
        *caution,
        "■ 予測を押し上げ(+)/押し下げ(-)している主な要因（ロジスティック回帰）",
        *[f"  {feature_ja(n)}: {v:+.3f}" for n, v in drivers],
        "",
        "■ 過去5年分のウォークフォワード検証（各250営業日×5回）",
        *[f"  {m}: 正解率 {r.accuracy:.1%} / AUC {r.auc:.3f} / 常に多数派を当てた場合 {r.baseline:.1%}"
          for m, r in summary.iterrows()],
        "",
        "■ 採用した相関市場（直近{}営業日、翌日の日経との相関）".format(CORR_WINDOW),
        *[f"  {r['name']:<14} 同日 {r.corr_same_day:+.2f} / 翌日 {r.corr_next_day:+.2f}"
          for _, r in corr_tbl[corr_tbl.selected].iterrows()],
        "",
        f"■ 実運用の成績: {int(done['hit'].sum())}/{len(done)} 的中" if len(done) else "■ 実運用の成績: まだ答え合わせ可能な予測なし",
        "",
        "※ 統計モデルによる参考情報であり、投資判断を保証するものではありません。",
    ]
    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "latest_report.txt").write_text(report, encoding="utf-8")
    corr_tbl.to_csv(OUT_DIR / "market_correlation.csv")
    wf.to_csv(OUT_DIR / "walk_forward.csv", index=False)
    (OUT_DIR / "latest.json").write_text(json.dumps(
        {"base_date": base_date, "judge_date": judge_day.strftime("%Y-%m-%d"),
         "target_date": target_day.strftime("%Y-%m-%d"), "prob_up": prob_up, "direction": direction,
         "model": best, "markets": sel, "subject": subject}, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_mail and not args.demo:
        send_mail(subject, report)


if __name__ == "__main__":
    main()
