#!/usr/bin/env bash
# 検証用: 外部サイトのページを取得して raw/ に保存（GitHub Actions 上で実行）
set -u
mkdir -p raw
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
while read -r name url; do
  [ -z "$name" ] && continue
  code=$(curl -sSL -A "$UA" -H "Accept-Language: ja,en;q=0.8" -o "raw/$name" -w "%{http_code}" "$url" || echo ERR)
  echo "$code $(wc -c < "raw/$name" 2>/dev/null) $name $url"
done < "${1:-urls.txt}"
