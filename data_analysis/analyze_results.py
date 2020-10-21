#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: analyze_results.py <result_data>
"""

from docopt import docopt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Useful way to run the algorithm over multiple files
#
# for file in `ls ~/Desktop/complete_results/*/*/*/*.csv | grep prediction.csv | sort`; do ./analyze_results.py $file; done;
#

if __name__ == "__main__":

	arguments = docopt(__doc__)
	file_path = arguments["<result_data>"]

	# Read the file
	results = pd.read_csv(file_path)

	# Calculate some metrics
	mse = mean_squared_error(results["incidence"].fillna(0), results["prediction"])
	r2 = r2_score(results["incidence"].fillna(0), results["prediction"])
	pcc = np.corrcoef(results["incidence"].fillna(0), results["prediction"], rowvar=False)[0][1]

	print(file_path + "," + str(mse) + "," + str(r2) + "," + str(pcc))

