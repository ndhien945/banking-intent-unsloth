#!/bin/bash

set -e

echo -e "\n1. Single sentence..."
python scripts/inference.py --message "Help! I just lost my wallet and my credit card is in it. What should I do?"
python scripts/inference.py --message "Why is the exchange rate applied to my recent purchase so bad?"

echo -e "\n2. Evaluation..."
python scripts/inference.py --evaluate sample_data/test.csv

echo -e "\nDone"