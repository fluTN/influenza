#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

year=''
month=''

for year in {2007..2016}; do
    for month in {01..12}; do

        if [ "$year" -eq "2007" -a "$month" -lt "12" ]; then continue; fi

        echo "[*] Making dir ${year}-${month}"
        mkdir -p "${year}-${month}"
    done
done
