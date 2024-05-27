import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap
import seaborn as sns

sys.path.insert(0, "../../src")

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
cmap = ListedColormap([sns.color_palette("Blues")[5], 'lightskyblue', sns.color_palette("rocket")[3],sns.color_palette("rocket")[2]])
# cmap = ListedColormap([sns.color_palette("Blues")[5], 'mediumseagreen', "orange" , "lightskyblue" ])
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)
