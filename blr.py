import numpy as np
import pandas as pd
"""
meta = {
    "name": {'age': 0, 'sex': 1, 'chest_pain': 2, 'rest_bp': 3, 'chol': 4, 'fast_bs_120': 5, 'rest_ecg': 6,
            'max_rate': 7, 'ex_angina': 8, 'oldpeak': 9, 'grad_peak': 10, 'num_vessels': 11, 'thal': 12},
    "type": ['real','bin','nom','real','real','bin','nom','real','bin','real','ord','real','nom'],
}

print(len(meta["name"]))
print(len(meta["type"]))
with open('data') as f:
    for row in f.readlines():
        print(row.split())
"""

data = pd.read_csv('data', sep=' ',
                    names=['age', 'sex', 'chest_pain', 'rest_bp',
                    'chol', 'fast_bs_120', 'rest_ecg', 'max_rate',
                    'ex_angina', 'oldpeak', 'grad_peak', 'num_vessels', 'thal'],
                    index_col=False)

print(data.head())
