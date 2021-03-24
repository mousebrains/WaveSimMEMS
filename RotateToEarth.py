#! /usr/bin/env python3
#
# Take the output of MkEllipse and rotate from wave coordinates to earth coordinates
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import numpy as np
import pandas as pd
import os.path

def rotate(a:pd.DataFrame) -> pd.DataFrame:
    # rotate to earth coordinates
    # hdg is the angle from true north eastward
    #  x -> Eastward
    #  y -> Northward
    #  z -> Vertical (This is the same in earth and wave coordinates

    hdg = np.radians(a.hdg) # Heading in radians
    shdg = np.sin(hdg)
    chdg = np.cos(hdg)

    df = pd.DataFrame()
    df["t"] = a["t"]
    df["hdg"] = a["hdg"]

    # Rotate position from wave to earth, +y -> North, +x -> East
    df["x"] = a.wx * shdg # Eastward
    df["y"] = a.wx * chdg # Northward
    df["z"] = a.wz        # Vertical

    # Rotate velocity from wave to earth
    df["vx"] = a.wvx * shdg # Eastward
    df["vy"] = a.wvx * chdg # Northward
    df["vz"] = a.wvz        # Vertical

    # Rotate acceleration from wave to earth
    df["ax"] = a.wax * shdg # Eastward
    df["ay"] = a.wax * chdg # Northward
    df["az"] = a.waz        # Vertical

    # Note, angular velocity and acceleration are in wy direction,
    # Rotate angular velocity from wave to earth
    df["omegax"] =  a.wOmega * chdg # Eastward
    df["omegay"] = -a.wOmega * shdg # Northward
    df["omegaz"] = np.zeros(a.wOmega.shape) # Vertical always zero

    # Rotate angular acceleration from wave to earth
    df["alphax"] =  a.wAlpha * chdg # Eastward
    df["alphay"] = -a.wAlpha * shdg # Northward
    df["alphaz"] = np.zeros(a.wAlpha.shape) # Vertical always zero

    return df

def saveCSV(fn:str, name:str, df:pd.DataFrame) -> None:
    (prefix, suffix) = os.path.splitext(fn)
    ofn = "{}.{}.earth.csv".format(prefix, name)
    df.to_csv(ofn, index=False)

def plotit(ax, df, x, y):
    ax.plot(df[x], df[y], "-")
    ax.grid()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_aspect(1)

if __name__ == "__main__":
    import argparse
    import MyLogger
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", metavar="fn.yml", help="YAML configuratioin file")
    parser.add_argument("wave", metavar="fn.csv", help="Output of mkEllipse to load")
    parser.add_argument("--save", action="store_true", help="Should CSV output files be generated?")
    parser.add_argument("--plot", action="store_true", help="Some diagnostic plots")
    MyLogger.addArgs(parser)
    args = parser.parse_args()

    logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

    a = pd.read_csv(args.wave)

    # Get the wave name from the wave csv filename, *.name.ellipse.csv
    parts = args.wave.split(".")
    name = parts[-3] # Wave name

    df = rotate(a)
    logger.info("Earth %s\n%s", name, df)

    if args.save:
        saveCSV(args.yaml, name, df)

    if args.plot:
        (fig, ax) = plt.subplots(3,2, figsize=(10,10))
        plotit(ax[0,0], df, "x", "z")
        plotit(ax[1,0], df, "y", "z")
        plotit(ax[2,0], df, "x", "y")
        plotit(ax[0,1], df, "vx", "vz")
        plotit(ax[1,1], df, "vy", "vz")
        plotit(ax[2,1], df, "vx", "vy")
        plt.show()
