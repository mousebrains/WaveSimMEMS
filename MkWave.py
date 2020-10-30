#! /usr/bin/env python3
#
# Generate a wave train specified by
#  data["depth"] the water depth in meters
#  data["duration"] sampling period in seconds
#  wave["period"]+-wave["periodSigma"] # Wave's period in seconds
#  wave["amplitude"]+-wave["amplitudeSigma"] # Wave's amplitude in meters
#  wave["heading"]+-wave["headingSigma"] # Wave's heading in degrees from true north
#
# The return values for each wave in the wave train are:
#  df["period"] Wave's period in seconds
#  df["amp"] Wave's amplitude in meters
#  df["hdg"] Wave's heading in in degrees from true north
#  df["lambda"] Wave's wave length in meters
#  df["spd"] Wave's phase speed in meters/sec
#  df["a"] and df["b"] Wave's ellipse major and minor radiuses in meters
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import numpy as np
import pandas as pd
import math
import WaveLength
import os.path

def mkWave(data:dict, wave:dict, rs:np.random.RandomState) -> dict:
    depth = data["depth"]
    duration = data["duration"]
    tMin = duration + wave["period"] # Make sure we have a whole extra period
    nWaves = math.ceil(duration / wave["period"])
    for cnt in range(5, 50, 5):
        period = rs.normal(wave["period"], wave["periodSigma"], size=(nWaves+cnt,))
        period[period <= 0] = period[period > 0].min()
        t = period.cumsum()
        if t.max() >= tMin: # Long enough, so keep
            period = period[t <= tMin]
            break

    df = pd.DataFrame({"period": period})
    df["amp"] = rs.normal(wave["amplitude"], wave["amplitudeSigma"], size=period.shape)
    df["hdg"] = rs.normal(wave["heading"], wave["headingSigma"], size=period.shape)
    df["lambda"] = WaveLength.waveLength(depth, period)
    df["spd"] = df["lambda"] / df["period"] # Phase speed in m/sec

    # See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
    # Particle Orbits at the surface, i.e. y=gliderDepth
    k = 2 * np.pi / df["lambda"] # Wave Number
    y = data["gliderDepth"]
    denom = np.sinh(k * depth)
    df["a"] = df["amp"] * np.cosh(k * (y + depth)) / denom # horizontal ellipse parameter
    df["b"] = df["amp"] * np.sinh(k * (y + depth)) / denom # vertical ellipse parameter
    return df

def saveCSV(fn:str, name:str, info:pd.DataFrame) -> None:
    (prefix, suffix) = os.path.splitext(fn)
    ofn = "{}.{}.info.csv".format(prefix, name)
    info.to_csv(ofn, index=False)

if __name__ == "__main__":
    import argparse
    import MyYAML
    import MyLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="+", metavar="fn.yml", help="YAML file(s) to load")
    parser.add_argument("--seed", type=int, help="Random number generator seed, 32 bit int")
    parser.add_argument("--save", action="store_true", help="Should CSV output files be generated?")
    MyLogger.addArgs(parser)
    args = parser.parse_args()

    logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

    for fn in args.yaml:
        data = MyYAML.load(fn, logger)
        rs = np.random.RandomState(seed=args.seed)
        seen = set()
        for wave in data["waves"]:
            name = wave["name"]
            for key in sorted(wave):
                if key == "name": continue
                logger.info("wave %s %s -> %s", name, key, wave[key])
            if name in seen:
                raise Exception("Duplicate wave name, {}, found in {}".format(name, fn))
            seen.add(name)
            info = mkWave(data, wave, rs)
            logger.info("Wave %s\n%s", name, info)
            if args.save:
                saveCSV(fn, name, info)
