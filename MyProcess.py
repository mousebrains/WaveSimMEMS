#
# Read in a YAML configuration file and
# generate a series of simulated waves.
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import argparse
import yaml
import logging
import os.path
import math
import numpy as np
import pandas as pd

def addArgs(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, help="Random seed, 32 bit integer")

def process(fn:str, args:argparse.ArgumentParser, logger:logging.Logger) -> bool:
    (prefix, suffix) = os.path.splitext(fn)
    logfn = prefix + ".log"
    csvfn = prefix + ".csv"
    ch = logging.FileHandler(logfn, mode="w")
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logger.addHandler(ch)
    try:
        rs = np.random.RandomState(seed=args.seed)
        __process(fn, csvfn, rs, logger)
        return True
    except:
        logger.exception("Error processing %s", fn)
        return False
    finally:
        logger.removeHandler(ch)

def __process(fn:str, csvfn:str, rs:np.random.RandomState, logger:logging.Logger) -> bool:
    data = __loadYAML(fn, logger)
    if data is None: return False

    dt = 1 / data["samplingRate"] # Time between observations
    t = np.arange(0, data["duration"] + dt, dt) # observation times [0, duration]

    for key in sorted(data):
        if key != "waves":
            logger.info("%s -> %s", key, data[key])

    waves = {}
    ellipses = {}
    pos = None
    for wave in data["waves"]:
        name = wave["name"]
        for key in sorted(wave):
            if key == "name": continue
            logger.info("wave %s %s -> %s", name, key, wave[key])
        if name in waves:
            raise Exception("Duplicate wave name, {}, found in {}".format(name, fn))
        waves[name] = __mkWave(data, wave, rs)
        # logger.info("Wave %s\n%s", name, waves[name])
        ellipses[name] = __mkEllipse(data["depth"], data["gliderDepth"], waves[name], t, rs)
        # logger.info("Ellipse %s\n%s", name, ellipses[name])
        a = ellipses[name][["t", "x", "y", "z"]].copy()
        if pos is None: # First wave
            pos = a
        else: # Add additional waves
            pos["x"] += a["x"]
            pos["y"] += a["y"]
            pos["z"] += a["z"]

    if pos is None:
        raise Exception("No waves found for {}".format(fn))

    dt = pos["t"].diff() # Time between samples in seconds

    # Linear acceleration in m/sec in each true direction
    pos["dxdt"] = pos["x"].diff() / dt
    pos["dydt"] = pos["y"].diff() / dt
    pos["dzdt"] = pos["z"].diff() / dt

    # Angular acceleration in radians/sec about each true axis
    # Use dot product x dot y = |x| |y| cos(theta)
    # and solve for theta to get dtheta/dt
    pos["dxydt"] = __mkAngularRate(pos["x"], pos["y"], dt)
    pos["dxzdt"]   = __mkAngularRate(pos["x"], pos["z"], dt)
    pos["dyzdt"]   = __mkAngularRate(pos["y"], pos["z"], dt)

    # TO BE ADDED
    # 
    # Glider response/transfer function
    # sensor noise
    #

    pos.to_csv(csvfn, index=False)

    return True

def __mkWave(data:dict, wave:dict, rs:np.random.RandomState) -> dict:
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
    df["lambda"] = __waveLength(depth, period)
    df["spd"] = df["lambda"] / df["period"] # Phase speed in m/sec

    # See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
    # Particle Orbits at the surface, i.e. y=gliderDepth
    k = 2 * np.pi / df["lambda"] # Wave Number
    y = data["gliderDepth"]
    denom = np.sinh(k * depth)
    df["a"] = df["amp"] * np.cosh(k * (y + depth)) / denom # horizontal ellipse parameter
    df["b"] = df["amp"] * np.sinh(k * (y + depth)) / denom # vertical ellipse parameter
    return df

def __waveLength(depth:float, period:np.array, threshold:float=0.01, maxIter:int=10) -> np.array:
    # See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
    # Phase speed
    # c = L/period = sqrt((g L)/(2 pi) * tanh(2 pi depth / L))
    # For deep water waves
    # L ~= g T^2 / 2pi
    # For shallow water waves
    # L ~= sqrt(g * depth) * period
    #
    # threshold is in meters
    #
    # Use Newton-Raphson method to converge to a solution
    # for:
    #  L^2 / period^2 = (gL/(2pi)) * tanh(2pi depth / L)
    #  L / period^2 = (g/(2pi)) * tanh(2pi depth / L)
    #  g period^2 / (2 pi) = L / tanh(2pi depth / L)
    #  2pi / (g * period^2) = tanh(2pi depth / L) / L
    #  lhs = rhs
    #  0 = lhs - rhs

    g = 9.80665 # gravitational acceleration m/sec^2
    twoPi = 2 * np.pi
    period2 = np.square(period) # Square of period
    twoPiDepth = twoPi * depth

    L = g * period2 / twoPi # Deep water approximation
    term = np.tanh(twoPi * depth / L) # see how close tanh is to one
    msk = abs(term) < 0.75 # Use shallow water approximation in these cases

    if msk.sum(): # Some shallow water apprixmations needed
        L[msk] = math.sqrt(g * depth) * period[msk] # Shallow water approximation

    lhs = twoPi / (g * period2)

    for cnt in range(maxIter): # Maximum number of iterations
        dL = (0.001 * L) / 2 # derivative step size
        xlhs = L - dL
        xrhs = L + dL
        y = lhs - np.tanh(twoPiDepth / L) / L
        ylhs = lhs - np.tanh(twoPiDepth / xlhs) / xlhs
        yrhs = lhs - np.tanh(twoPiDepth / xrhs) / xrhs
        dy = (ylhs - yrhs) / (xlhs - xrhs)
        dL = y / dy
        L -= dL
        if abs(dL).max() < threshold:
            break

    return L

def __mkEllipse(depth:float,
        gliderDepth:float,
        df:pd.DataFrame, 
        t:np.array, 
        rs:np.random.RandomState) -> pd.DataFrame:
    # See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
    # Particle Orbits at the surface, i.e. y==0

    w0 = rs.uniform(0, 2 * np.pi) # Initial random phase of wave
    dt0 = df["period"][0] * w0 / (2 * np.pi) # Initial random offset in time into first wave

    amp = df["amp"].to_numpy()
    k = 2 * np.pi / df["lambda"].to_numpy() # Wave Number
    omega = k * df["spd"].to_numpy() # angular frequency
    denom = np.sinh(k * depth)
    dx0 = -amp * np.cosh(k * (gliderDepth + depth)) / denom
    dy0 =  amp * np.sinh(k * (gliderDepth + depth)) / denom

    # Which wave to use at t with the random starting time offset
    indices = np.searchsorted(df["period"].cumsum(), t + dt0)
    indices[indices >= omega.size] = omega.size - 1

    a = pd.DataFrame({"t": t})

    term = -omega[indices] * (t + dt0)
    a["dx"] = dx0[indices] * np.sin(term)
    a["dy"] = dy0[indices] * np.cos(term)

    hdg = df["hdg"].to_numpy()[indices]
    a["hdg"] = hdg
    hdg = np.radians(hdg)

    # Now rotate by heading into true north coordinates
    #  x -> Eastward
    #  y -> Northward
    #  z -> Vertical
    #  hdg is clockwise angle from true north
    a["x"] = a["dx"] * np.sin(hdg) # Eastward direction
    a["y"] = a["dx"] * np.cos(hdg) # Northward direction
    a["z"] = a["dy"] # Vertical position

    return a

def __mkAngularRate(x:pd.DataFrame, y:pd.DataFrame, dt:np.array) -> pd.DataFrame:
    # Use dot product to get angle between consecutive time stamps
    #   cos(theta) = 
    #         (x[0] * x[1] + y[0] * y[1]) / 
    #         (sqrt(x[0]^2 + y[0]^2) * sqrt(x[1]^2 + y[1]^2))
    
    x = x.to_numpy()
    y = y.to_numpy()
    a = x[0:-1] * x[1:] + y[0:-1] * y[1:] # Dot product
    norm = np.sqrt(x*x + y * y) # length of each vector
    b = a / (norm[0:-1] * norm[1:]) # cos(theta)
    dtheta = np.arccos(b)
    dthetadt = dt.copy() 
    dthetadt[1:] = dtheta / dt[1:]
    return dthetadt

def __loadYAML(fn:str, logger:logging.Logger) -> dict:
    try:
        with open(fn, "r") as fp:
            lines = fp.read()
            logger.info("Processing %s\n%s", fn, lines)
            data = yaml.load(lines, Loader=yaml.SafeLoader)
            return data
    except Exception as e:
        logger.exception("Error loading %s", fn)
        raise e
