import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


COLORS = {
    'white':    '#FFFFFF',
    'black':    "#000000",
    'bkg':      "#404040",
    "red":      "#D07070",
    "green":    "#70D070",
    "blue":     "#5060FF",
    "cyan":     "#30A0A0",
    "magenta":  "#A030A0",
    "yellow":   "#A0A030",
    }

mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.size'] = 14.0
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['image.interpolation'] = 'nearest'

mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.5

mpl.rcParams['lines.markersize'] = 5.0
mpl.rcParams['lines.linewidth'] = 2.0

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.top'] = False
mpl.rcParams['xtick.labelbottom'] = True
mpl.rcParams['xtick.labeltop'] = False
mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = False
mpl.rcParams['ytick.labelleft'] = True
mpl.rcParams['ytick.labelright'] = False


cdict = {
    'red':   ((0.000, 0.0, 0.0),
              (0.999, 1.0, 1.0),
              (1.000, 1.0, 1.0)),
    'green': ((0.000, 0.0, 0.0),
              (0.999, 1.0, 0.7),
              (1.000, 0.7, 0.7)),
    'blue':  ((0.000, 0.0, 0.0),
              (0.999, 1.0, 0.7),
              (1.000, 0.7, 0.7)),
    }

gray_saturated = LinearSegmentedColormap("GraySaturated", cdict)
plt.register_cmap(cmap=gray_saturated)

