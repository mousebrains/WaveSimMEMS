#
# Read in a YAML configuration file and
# generate a series of simulated waves.
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import argparse
import MyYAML
import logging
import os.path
import numpy as np
import MkWaveTrain
import MkEllipse

def addArgs(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, help="Random seed, 32 bit integer")

def process(fn:str, args:argparse.ArgumentParser, logger:logging.Logger) -> bool:
    (prefix, suffix) = os.path.splitext(fn)
    logfn = prefix + ".log"
    ch = logging.FileHandler(logfn, mode="w")
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logger.addHandler(ch)
    try:
        rs = np.random.RandomState(seed=args.seed)
        __process(fn, prefix, rs, logger)
        return True
    except:
        logger.exception("Error processing %s", fn)
        return False
    finally:
        logger.removeHandler(ch)

def __process(fn:str, prefix:str, rs:np.random.RandomState, logger:logging.Logger) -> bool:
    data = MyYAML.load(fn, logger)
    if data is None: return False

    dt = 1 / data["samplingRate"] # Time between observations
    t = np.arange(0, data["duration"] + dt, dt) # observation times [0, duration]

    for key in sorted(data):
        if key != "waves":
            logger.info("%s -> %s", key, data[key])

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
        info = MkWaveTrain.mkTrain(data, wave, rs)
        MkWaveTrain.saveCSV(fn, name, info)
        ellipse = MkEllipse.mkEllipse(data["depth"], data["gliderDepth"], info, t, rs)
        MkEllipse.saveCSV(fn, name, ellipse)
        ellipse.to_csv("{}.{}.ellipse.csv".format(prefix, name), index=False)
        if pos is None: # First wave
            keys = ["t"]
            for key in sorted(ellipse):
                if (key not in ["t", "hdg"]) and (key[0] != "w"):
                    keys.append(key)
            pos = ellipse[keys].copy()
        else: # Add additional waves
            for key in pos:
                if key != "t":
                    pos[key] += ellipse[key]

    if pos is None:
        raise Exception("No waves found for {}".format(fn))

    # TO BE ADDED
    # 
    # Glider response/transfer function
    # sensor noise
    #

    pos.to_csv("{}.csv".format(prefix), index=False)

    return True
