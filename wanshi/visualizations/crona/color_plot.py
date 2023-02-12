from wanshi.visualizations.crona.met_brewer.palettes import met_brew

colors = met_brew(name="Renoir" ,brew_type="continuous")
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 1, 10)
for i, color in enumerate(colors, start=1):
    plt.plot(x, i * x + i, color=color, label='$y = {i}x + {i}$'.format(i=i))
plt.legend(loc='best')
plt.show()
