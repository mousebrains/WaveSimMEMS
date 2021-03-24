# Given the results of Analysis.py, plot a polar directional spectrum
#
# Mar-2021, Pat Welch, pat@mousebrains.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors

def plotit(data:pd.DataFrame, threshold:float=None) -> None:
    C11 = data.Czz.to_numpy()
    a0 = data.a0.to_numpy()
    a1 = data.a1.to_numpy()
    b1 = data.b1.to_numpy()
    a2 = data.a2.to_numpy()
    b2 = data.b2.to_numpy()

    # Grid points to compute and plot at
    freq = data.freq.to_numpy() # n rows
    theta = np.radians(np.arange(0, 360+1, 5)) # m rows, with the first and last replicated

    (freqG, thetaG) = np.meshgrid(freq, theta) # each array is mxn

    # From eqn 22 in Gorman for just the first two terms

    D = 1/(2*np.pi) + (
            a1 * np.cos(  thetaG) + b1 * np.sin(  thetaG) +
            a2 * np.cos(2*thetaG) + b2 * np.sin(2*thetaG)
            ) / np.pi

    # Directional spectra
    S = C11 * D # N.B. a1, b1, a2, and b2 have already been divided by C11/Czz

    if True:
        (fig, ax) = plt.subplots(1, figsize=(10,10), subplot_kw=dict(polar=True))
        polarPlot(fig, ax, freqG, np.degrees(thetaG), S, nLevels=256,
                threshold=threshold, periodMax=-10, periodMin=3)

    if False:
        xlim = (1/33, 1/3)
        (fig, ax) = plt.subplots(3,2, figsize=(10,10))
        plotLinear(ax[0,0], freq, C11, "C11", xlim)
        plotLinear(ax[1,0], freq, a1, "a1", xlim)
        plotLinear(ax[1,1], freq, b1, "b1", xlim)
        plotLinear(ax[2,0], freq, a2, "a2", xlim)
        plotLinear(ax[2,1], freq, b2, "b2", xlim)
    plt.show()

def plotLinear(ax, f, y, label, xlim:tuple=None):
    msk = np.logical_and(f >= 1/33, f <= 1/2) 
    ax.plot(f[msk], y[msk], '-')
    ax.set_ylabel(label)
    ax.grid()
    if xlim is not None:
        ax.set_xlim(xlim)

def polarPlot(fig, ax, freq:np.array, theta:np.array, z:np.array,
        nLevels:int, threshold:float=1e-4,
        periodMin:float=3, periodMax:float=-10,
        rTicks:tuple=(20, 12, 8, 6, 4)) -> None:

    zMax = z.max()
    levelsAbove = np.linspace(threshold, zMax, nLevels) # levels above threshold
    dAbove = np.mean(np.diff(levelsAbove)) # Mean difference between levels
    levelsBelow = np.arange(0, threshold, dAbove)
    levels = np.append(levelsBelow, levelsAbove)

    jet = cm.get_cmap("jet", nLevels) # Blue to red above threshold
    azure = mcolors.ListedColormap(["azure"]) # Below threshold color
    cmap = np.row_stack((
        azure(np.arange(levelsBelow.size)),
        jet(np.arange(nLevels)))) # Color map with azure below threshold

    ax.tick_params(axis="x", direction="in", pad=-18) # Theta labels inside plot

    axColor = ax.contourf(
            np.radians(theta), freq, z,
            levels=levels, # Level contours
            colors=cmap)

    ax.set_theta_zero_location("N") # North at top
    ax.set_theta_direction("clockwise") # East/90 to the right

    ax.set_rgrids(radii=1/np.array(rTicks),
            labels=map(lambda x: "{} s".format(x), rTicks),
            angle=0, # along the North axis
            color="m") # Color of the labels
    ax.set_thetagrids(np.arange(30, 360+1, 30), labels=None)

    ax.set_rlim(
            bottom=1/periodMax,
            top=1/periodMin) # Reverse limits, small period on outside

    # Display the colorbar at the bottom
    cbar = fig.colorbar(axColor, orientation="horizontal",
            ticks=(0, zMax/2, zMax))
    cbar.set_label("Energy Density (m*m/Hz/deg)",
            fontsize=16, fontweight="bold",
            loc="left")
