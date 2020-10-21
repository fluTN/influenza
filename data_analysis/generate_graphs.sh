#!/bin/bash
# Generate the graphs used inside the papers (the graphs which shows to the influenza
# season estimation)
# Author: Giovanni De Toni (giovanni.det@gmail.com)

country=("italy" "germany" "netherlands" "belgium")

for c in ${country[@]}
do
  mkdir -p graphs/$c
  python3 compare_models.py new_data cyclerank pagerank --no-graph --not-dual --country=$c --start-year=2015-42 --end-year=2019-15 --basedir ../models/complete_results_standardized_2/
  python3 compare_models.py pageviews cyclerank_pageviews pagerank_pageviews --not-dual --no-graph --country=$c --start-year=2015-42 --end-year=2019-15 --basedir ../models/complete_results_standardized_2/
  mv 2015-42_2019-15_compare_results_new_data_${c}.png new_data_${c}.png
  mv new_data_${c}.png graphs/$c
  mv 2015-42_2019-15_compare_results_pageviews_${c}.png pageviews_${c}.png
  mv pageviews_${c}.png graphs/$c

  rm 2015-42_2019-15_compare_results_new_data_${c}.csv
  rm 2015-42_2019-15_compare_results_pageviews_${c}.csv

done
