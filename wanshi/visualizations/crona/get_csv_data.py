
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv_path = 'META-AI_ Guideline Items - Tabellenblatt.csv'
core_data = 'core data.csv'

df_core_data = pd.read_csv(core_data, sep=',', header=0, index_col=0, encoding='utf-8', engine='python', error_bad_lines=False)
# get the first column and put into dict
first_column_core = df_core_data.iloc[:,0]
first_column_core_keys = first_column_core
print(first_column_core_keys)

first_column_core = first_column_core.values

first_column_core = [x for x in first_column_core if str(x) != 'nan']
print(first_column_core)
first_column_core_enu = enumerate(first_column_core)
for i in first_column_core_enu:
    print(i)
print(df_core_data.groupby('cata')['test'].count())

# guide_line_group = {'Clinical Rationale': 7, 'Data': 11, 'Model Training and Validation': 9, 'Critical Appraisal': 3, 'Ethics and Reproducibility': 7}
# first_column_core = df_core_data.iloc[:,0]
# print(first_column_core)
print('------------------')
for cri in range(21):
    #print(cri)
    for item in range((37)):
        #print(item)
        print(df_core_data.iloc[item, cri+4])