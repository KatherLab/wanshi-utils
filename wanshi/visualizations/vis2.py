import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the csv file
df = pd.read_csv('/home/jeff/PycharmProjects/wanshi-utils/wanshi/visualizations/crona/all.csv')



# Replace commas in numbers with dots to make them floats
df = df.replace(',', '.', regex=True)

# Cast all percentages to float
for col in df.columns[2:]:
    df[col] = df[col].astype(float)

# Define the groups
groups = {
    'Clinical Rationale': [
        'Topic', 'Study Design', 'Prediction Problem', 'Clinical Setting',
        'Rationale', 'Existing AI and Statistical Models', 'State-of-the-art'
    ],
    'Data': [
        'Data Sources Types and Structure', 'Data Selection', 'Data Preprocessing',
        'Labeling of Input Data', 'Rater Variability', 'Data Processing Location',
        'De-Identification', 'Data Dictionary', 'Data Leakage', 'Representativeness',
        'Basic Statistics of the Dataset'
    ],
    'Model Training and Validation': [
        'Type of Prediction Model', 'Model Development', 'Model Validation',
        'Model Interpretability', 'Model Performance and Interpretation',
        'Computational Cost', 'Statistical Methods', 'Performance Errors', 'Over-/Underfitting'
    ],
    'Critical Appraisal': [
        'Clinical Implications and Practical Value', 'Translation', 'Limitations'
    ],
    'Ethics and Reproducibility': [
        'Data Publication', 'Code Publication', 'AI Intervention Publication',
        'Future Updates', 'Ethical Statement', 'Equity and Access', 'Legal and Regulatory Aspects'
    ]
}

group_dict = {item: idx for idx, items in enumerate(groups.values()) for item in items}
group_colors = plt.cm.get_cmap('hsv', len(groups))
print(group_dict)
# Assign group ids to each row
df['group'] = [group_dict[item] for item in df['Guideline Item']]

# Create group color dataframe
group_color_df = pd.DataFrame(df['group'], columns=['group'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 20), gridspec_kw=dict(width_ratios=[1,20]))

# Define colormap in shades of gray
cmap = sns.light_palette("gray", as_cmap=True)

# Create colorbar
sns.heatmap(group_color_df, cmap=group_colors, cbar=False, yticklabels=False, ax=ax1)

# Create main heatmap
sns.heatmap(df.iloc[:, 2:-1], cmap=cmap, linewidths=1, linecolor='white', yticklabels=True, cbar=True, ax=ax2)
#revverse x and y axis
#ax2.invert_yaxis()
plt.show()