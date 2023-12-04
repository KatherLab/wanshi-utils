import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_file='percentage2.csv'
df = pd.read_csv(csv_file, sep=',', header=0, index_col=0)

# Convert percentage strings to floats for seaborn to plot correctly
df['% Item included in high-level consensus guidelines (Y/P)'] = df['% Item included in high-level consensus guidelines (Y/P)'].str.replace(',','.').astype('float')
df['% Item included in intermediate-level consensus guidelines (Y/P)'] = df['% Item included in intermediate-level consensus guidelines (Y/P)'].str.replace(',','.').astype('float')
df['% Item included in low-level consensus guidelines (Y/P)'] = df['% Item included in low-level consensus guidelines (Y/P)'].str.replace(',','.').astype('float')
df['% Item included in all guidelines (Y/P)'] = df['% Item included in all guidelines (Y/P)'].str.replace(',','.').astype('float')
print(df['% Item included in high-level consensus guidelines (Y/P)'])
# Define a size for the plot
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(df, annot=True, fmt=".2f", cmap='YlGnBu')

# Show the plot
plt.show()
