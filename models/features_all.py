#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter the features that appear in all trained models.

Usage:
  features_all.py [ --top=TOP_N ] FILE ...
  features_all.py (-h | --help)
  features_all.py --version

Options:
  -top TOP_N    Number of top N entries to show, -1 is all of them [default:10].
  -h --help     Show this screen.
  --version     Show version.

"""

import os
import csv
from docopt import docopt
from pprint import pprint

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')
    # print(arguments)

    files_dict = dict()
    for infile in arguments['FILE']:

        # first four letters of filename are the year
        #year = int(os.path.basename(infile)[0:4])

        with open(infile, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) # Skip header file
            for el in reader:
                year = el[0][5:9]
                if files_dict.get(year, None) is None:
                    files_dict[year] = dict([(el[2], float(el[3]))])
                else:
                    files_dict[year][el[2]]=float(el[3])

    terms = dict()
    for year, data in files_dict.items():
        for k, v in data.items():
            if k == 'mse': continue
            if terms.get(k, None) is None:
                terms[k] = [v]
            else:
                terms[k].append(v)

    terms_count = dict((k,len(v)) for k,v in terms.items())
    terms_avg = dict((k,sum(v)/float(len(v))) for k,v in terms.items())

    top_n = int(arguments['--top'] or 10)
    _terms_avg_top = sorted(sorted(terms_avg.items(),
                                     key=lambda value: value[0]),
                            key=lambda value: value[1],
                            reverse=True
                            )
    _terms_count_top = sorted(sorted(terms_count.items(),
                                     key=lambda value: value[0]),
                              key=lambda value: value[1],
                              reverse=True
                              )

    terms_avg_top = _terms_avg_top[0:top_n]
    terms_count_top = _terms_count_top[0:top_n]
    
    if top_n < 0:
        terms_avg_top = terms_avg_top[:]
        terms_count_top = terms_count_top[:]

    # import ipdb; ipdb.set_trace()
    pprint('--- terms_count_top ---')
    pprint(terms_count_top)
    pprint('--- terms_avg_top ---')
    pprint(terms_avg_top)
