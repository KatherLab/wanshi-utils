
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv_path = 'META-AI_ Guideline Items - Tabellenblatt.csv'
core_data = 'sup_data.csv'

df_core_data = pd.read_csv(core_data, sep=',')
# transpose the data
df_core_data = df_core_data.T
# print all the data in the first column
print(df_core_data.iloc[:,0])
# print all the data in the first row
print(df_core_data.iloc[0,:])
