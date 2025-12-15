#!/bin/bash

python ./data_cleaning.py
python ./split_data_vYOLO.py
python ./print_distribution_fold.py