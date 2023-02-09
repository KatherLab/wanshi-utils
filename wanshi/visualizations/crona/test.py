# read csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv_path = 'META-AI_ Guideline Items - Tabellenblatt.csv'
import csv

# read csv with pandas
df = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
#get the first row
first_row = df.iloc[0]
# get the first column
first_column = df.iloc[:,0]
critirian_names = df.columns.values[4:12]
print(critirian_names)
item_names = first_column.values
# remove nan from list
item_names = [x for x in item_names if str(x) != 'nan']

print(item_names)
print(first_row.keys())
sectors = {"Guideline Item": len(item_names), "Critirian": len(critirian_names)}
dict = {"Guideline Item": (item_names), "Critirian": (critirian_names)}
from pycirclize import Circos
from pycirclize.utils import ColorCycler
circos = Circos(sectors, space=5)
for sector in circos.sectors:
    print(sector.name)
    track1 = sector.add_track((90, 100))
    track1.axis()
    print(int(track1.size))



    pos_list = list(range(0, int(track1.size)))
    # add 0.5 to all positions
    pos_list = [x + 0.5 for x in pos_list]
    print(len(pos_list))
    # Plot rect & text (style1)
    for i in range(int(track1.size)):
        start, end = i, i + 1
        track1.rect(start, end, fc=ColorCycler())
    track1.xticks(
        pos_list,
        dict[sector.name],
        outer=True,
        tick_length=0,
        label_margin=2,
        label_orientation="vertical",
    )
    # Plot rect & text (style2)
    track2 = sector.add_track((80, 89))
    for i in range(int(track2.size)):
        start, end = i, i + 1
        track2.rect(start, end, fc=ColorCycler(), ec="white", lw=1)

critirian_link_list = enumerate(critirian_names)
# transfer item_names from numpy array to list
item_names_list = enumerate(item_names)
# get row and column number of the dataframe when the value is Y
tuple_list = []
for cri in range(len(critirian_names)):
    for item in range(len(item_names)):
        if df.iloc[item, cri+4] == "Y":
            print(item, cri)
            tuple_list.append((item, cri))

for i in range(len(tuple_list)):
    circos.link(("Guideline Item", tuple_list[i][0], tuple_list[i][0]+1), ("Critirian", tuple_list[i][1], tuple_list[i][1]+1),alpha=0.1)

fig = circos.plotfig()
fig.show()