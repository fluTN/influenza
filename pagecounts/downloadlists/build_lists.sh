#!/usr/bin/env bash
#
# Generate files to be used with the download.sh script
#
# Written by Giovanni De Toni <giovanni.det@gmail.com>

start_year=""
end_year=""
eval "$(docopts -V - -h - : "$@" <<EOF
Usage: build_lists.sh [options] <start_year> <end_year>

    -h, --help                  Show this help message and exits.
    --version                   Print version and copyright information.
----
build_lists.sh 0.1.0
copyright (c) 2017 Giovanni De Toni
MIT License
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
EOF
)"

set -euo pipefail
IFS=$'\n\t'

if [ "$start_year" -le "$end_year" ]; then
    echo "[ERROR] Start year must be greater than end year."
    exit 1
fi

for year in $(seq "$start_year" "$end_year"); do
    for month in {01..12}; do
        if [ -f "../sizes/$year-$month.txt" -a ! -f "./$year-$month.txt" ]; then
            ./make_lists.sh ../sizes/$year-$month.txt
        fi
    done
done
