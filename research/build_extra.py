#!/usr/bin/env python3
"""
検証用: JPX の週次データを取得して CSV にまとめる（GitHub Actions 上で実行）
  - 投資部門別売買状況（株式・金額・週次、二市場合計）: 海外投資家・個人の差引（買い−売り）
  - 信用取引現在高 過去推移表: 売残・買残（金額）
出力: extra/investor_weekly.csv, extra/margin_weekly.csv
"""
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.jpx.co.jp"
INV = BASE + "/markets/statistics-equities/investor-type/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
OUT = Path("extra")
OUT.mkdir(exist_ok=True)
S = requests.Session()
S.headers.update(UA)


def get(url, binary=False):
    for k in range(4):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                return r.content if binary else r.content.decode("utf-8", "ignore")
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


# ------------------------------------------------------------ 投資部門別
def list_investor_files():
    pages = [INV + "index.html"] + [INV + f"00-00-archives-{i:02d}.html" for i in range(0, 12)]
    links = set()
    for p in pages:
        try:
            h = get(p)
        except RuntimeError:
            continue
        links |= set(re.findall(r'href="(/markets/statistics-equities/investor-type/[^"]+/'
                                r'(?:stock_val_1_\d{6}\.xls|stock_1_w_\d{8}_\d{8}\.xlsx))"', h))
    return sorted(links)


def cell_text(df, i, j):
    v = df.iat[i, j]
    return "" if pd.isna(v) else str(v)


def parse_old(content):
    """旧形式（〜2022年頃）: シート 'Tokyo & Nagoya'、行=投資部門、列=前週/今週ブロック"""
    xl = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    name = next((n for n in xl if "Nagoya" in n), list(xl)[-1])
    d = xl[name]
    title = " ".join(cell_text(d, i, 0) for i in range(6))
    m = re.search(r"(\d{4})年\s*(\d{1,2})月.*?\(\s*(\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,2})\s*\)", title)
    if not m:
        raise ValueError("title not found")
    year, _, sm, sd, em, ed = map(int, m.groups())
    week_end = pd.Timestamp(year, em, ed)
    # 今週ブロックの見出し列を探す
    col = None
    for i in range(6, 16):
        for j in range(d.shape[1]):
            t = cell_text(d, i, j)
            mm = re.match(r"\s*(\d{1,2})/(\d{1,2})\s*～\s*(\d{1,2})/(\d{1,2})", t)
            if mm and (int(mm.group(3)), int(mm.group(4))) == (em, ed):
                col = j
        if col is not None:
            break
    if col is None:
        raise ValueError("week column not found")
    vcol = col + 1

    def block(label):
        for i in range(d.shape[0]):
            if label in cell_text(d, i, 0).replace("　", "").replace(" ", ""):
                return num(d.iat[i, vcol]), num(d.iat[i + 1, vcol]), num(d.iat[i + 2, vcol])
        raise ValueError(label)

    fs, fb, ft = block("海外投資家")
    is_, ib, it = block("個人")
    ts, tb, tt = block("総計")
    return dict(week_end=week_end, foreign_net=fb - fs, foreign_total=ft,
                indiv_net=ib - is_, indiv_total=it, total=tt)


def parse_new(content, fname):
    """新形式（2022年〜）: 行=市場、列=投資部門ごとに 売/買/差引/合計"""
    d = pd.read_excel(io.BytesIO(content), header=None)
    week_end = pd.Timestamp(re.search(r"_(\d{8})\.xlsx$", fname).group(1))
    # 大分類の見出し行（"海外投資家" を含む行）から各部門の列範囲を求める
    hrow = next(i for i in range(0, 7)
                if any("海外投資家" in cell_text(d, i, j) for j in range(d.shape[1])))
    starts = sorted(j for j in range(3, d.shape[1]) if cell_text(d, hrow, j).strip())
    span = {}
    for k, j in enumerate(starts):
        t = cell_text(d, hrow, j).replace("　", "").replace(" ", "")
        end = starts[k + 1] if k + 1 < len(starts) else d.shape[1]
        for key, lab in (("foreign", "海外投資家"), ("indiv", "個人")):
            if t.startswith(lab) and key not in span:
                span[key] = (j, end)
    hdr = span
    row = None
    for i in range(d.shape[0]):
        if "二市場" in cell_text(d, i, 1) or "Tokyo & N" in cell_text(d, i, 1):
            row = i + 1 if "金額" not in cell_text(d, i, 2) else i
            if "金額" not in cell_text(d, row, 2):
                row = i
            break
    if row is None or "foreign" not in hdr:
        raise ValueError(f"layout {hdr} row={row}")

    def agg(key):  # 範囲内の 4列ブロック（売, 買, 差引, 合計）を合算
        a, b = hdr[key]
        s = sum(num(d.iat[row, j]) or 0 for j in range(a, b, 4))
        p = sum(num(d.iat[row, j + 1]) or 0 for j in range(a, b, 4))
        t = sum(num(d.iat[row, j + 3]) or 0 for j in range(a, b, 4))
        return s, p, t

    fs, fb, ft = agg("foreign")
    is_, ib, it = agg("indiv")
    return dict(week_end=week_end, foreign_net=fb - fs, foreign_total=ft,
                indiv_net=ib - is_, indiv_total=it, total=None)


def build_investor():
    links = list_investor_files()
    print(f"investor files: {len(links)}")
    rows, errs = [], 0
    for k, href in enumerate(links):
        fname = href.rsplit("/", 1)[1]
        try:
            c = get(BASE + href, binary=True)
            r = parse_new(c, fname) if fname.endswith(".xlsx") else parse_old(c)
            r["file"] = fname
            rows.append(r)
        except Exception as e:
            errs += 1
            print(f"[warn] {fname}: {e}")
        time.sleep(0.3)
    df = pd.DataFrame(rows).sort_values("week_end").drop_duplicates("week_end", keep="last")
    df.to_csv(OUT / "investor_weekly.csv", index=False)
    print(f"investor rows={len(df)} errors={errs} range={df.week_end.min()}..{df.week_end.max()}")


# ------------------------------------------------------------ 信用残
def build_margin():
    h = get(BASE + "/markets/statistics-equities/margin/05.html")
    href = re.findall(r'href="([^"]+_mtgk\.xls)"', h)[0]
    d = pd.read_excel(io.BytesIO(get(BASE + href, binary=True)), header=None)
    start = d.index[d[0].astype(str).str.contains("東京・名古屋|Tokyo & Nagoya")][0]
    rows = []
    for i in range(start + 1, d.shape[0]):
        v = d.iat[i, 0]
        if isinstance(v, str) and v.startswith("["):
            break
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(dict(week_end=dt, short_value=num(d.iat[i, 2]), long_value=num(d.iat[i, 4])))
    # 直近数週（過去推移表に未反映の分）を mtcurrent から補う
    h4 = get(BASE + "/markets/statistics-equities/margin/04.html")
    for href in sorted(set(re.findall(r'href="([^"]+_mtcurrent\.xlsx)"', h4))):
        try:
            c = pd.read_excel(io.BytesIO(get(BASE + href, binary=True)), header=None)
            wk = pd.Timestamp(re.search(r"/(\d{8})_mtcurrent", href).group(1))
            for i in range(c.shape[0]):
                if re.search(r"合計|Total", " ".join(str(x) for x in c.iloc[i, :3])) and i > 5:
                    vals = [num(x) for x in c.iloc[i].tolist()]
                    print("mtcurrent", wk.date(), [v for v in vals if v is not None][:8])
                    break
        except Exception as e:
            print("[warn] mtcurrent", href, e)
    df = pd.DataFrame(rows).drop_duplicates("week_end").sort_values("week_end")
    df.to_csv(OUT / "margin_weekly.csv", index=False)
    print(f"margin rows={len(df)} range={df.week_end.min()}..{df.week_end.max()}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["margin", "investor"]
    if "margin" in what:
        build_margin()
    if "investor" in what:
        build_investor()
