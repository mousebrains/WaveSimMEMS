#
# Read in a YAML configuration file and
# generate a series of simulated waves.
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import argparse
import MyYAML
import logging
import time
import os.path
import numpy as np
import MkWaveTrain
import MkEllipse
import RotateToEarth
import Analysis
import pandas as pd

def addArgs(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=int(time.time()),
            help="Random seed, 32 bit integer")

def process(fn:str, args:argparse.ArgumentParser, logger:logging.Logger) -> pd.DataFrame:
    (prefix, suffix) = os.path.splitext(fn) # Strip off yaml file suffix
    logfn = prefix + ".log" # Make a new log filename for this process
    ch = logging.FileHandler(logfn, mode="w")
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logger.addHandler(ch)
    logger.info("Seed=%s", args.seed)
    np.random.seed(args.seed)
    try:
        return __process(fn, prefix, logger)
    except:
        logger.exception("Error processing %s", fn)
        return None
    finally:
        logger.removeHandler(ch)

def __process(fn:str, prefix:str, logger:logging.Logger) -> pd.DataFrame:
    data = MyYAML.load(fn, logger)
    if data is None: return False

    for key in sorted(data):
        if key != "waves":
            logger.info("%s -> %s", key, data[key])

    dt = 1 / data["samplingRate"] # Time between model steps
    t = np.arange(0, data["duration"] + dt/2, dt) # model times [0, duration]

    pos = None
    seen = set()
    for wave in data["waves"]:
        name = wave["name"]
        for key in sorted(wave):
            if key == "name": continue
            logger.info("wave %s %s -> %s", name, key, wave[key])
        if name in seen:
            raise Exception("Duplicate wave name, {}, found in {}".format(name, fn))
        seen.add(name)
        info = MkWaveTrain.mkTrain(data, wave)
        MkWaveTrain.saveCSV(fn, name, info)
        ellipse = MkEllipse.mkEllipse(data["depth"], data["gliderDepth"], info, t)
        MkEllipse.saveCSV(fn, name, ellipse)
        earth = RotateToEarth.rotate(ellipse)
        RotateToEarth.saveCSV(fn, name, earth)
        if pos is None: # First wave
            pos = earth[earth.columns[earth.columns != "hdg"]].copy()
        else: # Add additional waves
            cols = earth.columns[earth.columns != "hdg"]
            cols = cols[cols != "t"]
            pos[cols] += earth[cols]

    if pos is None:
        raise Exception("No waves found for {}".format(fn))

    # TO BE ADDED
    # 
    # Glider response/transfer function
    # sensor noise
    #

    pos.to_csv("{}.csv".format(prefix), index=False)

    (info, ndInfo) = Analysis.analyze(fn, data, pos, data["depth"], logger, 
            nPerSegment=data["nPerSegment"] if "nPerSegment" in data else None,
            window=data["windowType"] if "windowType" in data else "boxcar")
    Analysis.saveCSV(fn, info, "analysis")
    Analysis.saveCSV(fn, ndInfo, "ndInfo")
    
    return info
