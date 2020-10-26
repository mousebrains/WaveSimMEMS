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
    data = __loadYAML(fn, logger)
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
        info = __mkWave(data, wave, rs)
        # logger.info("Wave %s\n%s", name, info)
        info.to_csv("{}.{}.info.csv".format(prefix, name), index=False)
        ellipse = __mkEllipse(data["depth"], data["gliderDepth"], info, t, rs)
        # logger.info("Ellipse %s\n%s", name, ellipse)
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
    # Particle Orbits at the surface
    # N.B. x in the MIT lecture notes is along the wave's travel direction
    #      y is the vertical direction (I will translate y to z here)

    # Start the wave train with a random time/phase offset
    dt0 = df["period"][0] * rs.uniform(0, 1) # Random time offset, [0,period)

    amp = df["amp"].to_numpy()
    k = 2 * np.pi / df["lambda"].to_numpy() # Wave Number
    omega = 2 * np.pi / df["period"].to_numpy() # Angular frequency
    denom = np.sinh(k * depth)
    zbar = gliderDepth  # this is <z> or <y> in the lecture notes
    x0 = -amp * np.cosh(k * (depth + zbar)) / denom
    z0 =  amp * np.sinh(k * (depth + zbar)) / denom

    # Which wave to use at t with the random starting time offset
    # This will transition between waves at a phase of zero
    indices = np.searchsorted(df["period"].cumsum(), t + dt0)
    indices[indices >= omega.size] = omega.size - 1

    x0 = x0[indices] # One to many mapping wave to time
    z0 = z0[indices]

    a = pd.DataFrame({"t": t})

    # The w prefix indicates in wave coordinates, i.e. wx -> Along wave, wz -> vertical
    # sin/cos(-w(t+dt))
    w    = -omega[indices] # Negative angular frequency at each time t
    term = w * (t + dt0)   # sine/cosine argument
    sterm = np.sin(term)
    cterm = np.cos(term)

    # The position at time t
    a["wx"] = x0 * sterm # Horizontal position along the wave direction
    a["wz"] = z0 * cterm # Vertical position

    # the velocity at time t, i.e. first derivative wrt t
    a["wvx"] =  x0 * w * cterm # d(wx)/dt
    a["wvz"] = -z0 * w * sterm # d(wz)/dt

    # the acceleration at time t, the second derivative wrt t
    a["wax"] = -x0 * w * w * sterm # d^2(wx)/d^2t
    a["waz"] = -z0 * w * w * cterm # d^2(wz)/d^2t

    # the angular velocity
    # omega = (r cross v) / |r|^2
    #       = perpendicular velocity over |r|
    # r = (wx,  0, wz)
    # v = (wvx, 0, wvz)
    # |r| = sqrt(wx^2 + wz^2)
    # vPerp = |r cross v| / |r|
    r = np.sqrt(np.square(a["wx"]) + np.square(a["wz"])) # radial distance
    a["wr"] = r # radial distance
    a["wvPerp"] = (a["wz"] * a["wvx"] - a["wx"] * a["wvz"]) / r # (r cross v) / r
    a["wOmega"] = a["wvPerp"] / r # Angular velocity in wave space in wy direction

    # the angular accceleration
    # a = (wax, 0, waz)
    # alpha = d((r cross v) / |r|^2) / dt
    #       = (r cross a)/|r|^2 - 2/|r| omega d|r|/dt
    # d|r|/dt = d((wx^2 + wz^2)^1/2)/dt
    #         = 1/2 (wx^2 + wz^2)^(-1/2) d(wx^2 + wz^2)/dt
    #         = 1/(2r) (2 wx d(wx)/dt + 2 wz d(wz)/dt)
    #         = (wx wvx + wz wvz) / r
    # since 2D I only need to calculate y component
    alpha0 = (a["wz"] * a["wax"] - a["wx"] * a["waz"]) / np.square(r) # (r crossa) / r^2
    drdt = (a["wx"] * a["wvx"] + a["wz"] * a["wvz"]) / r # dr/dt
    alpha1 = -2/r * a["wOmega"] * drdt # 2/|r| omega d|r|/dt
    a["wAlpha"] = alpha0 + alpha1 # angular acceleration in wy direction

    # Now rotate to earth coordinates
    # hdg is the angle from true north eastward
    #  x -> Eastward
    #  y -> Northward
    #  z -> Vertical (This is the same in earth and wave coordinates

    a["hdg"] = df["hdg"].to_numpy()[indices] # wave direction in degrees at time t
    hdg = np.radians(a["hdg"]) # Heading at t in radians
    shdg = np.sin(hdg)
    chdg = np.cos(hdg)

    # Rotate position from wave to earth
    a["x"]      = a["wx"] * shdg # Eastward
    a["y"]      = a["wx"] * chdg # Northward
    a["z"]      = a["wz"]        # Vertical

    # Rotate velocity from wave to earth
    a["vx"]   = a["wvx"] * shdg # Eastward
    a["vy"]   = a["wvx"] * chdg # Northward
    a["vz"]   = a["wvz"] * chdg # Vertical

    # Rotate acceleration from wave to earth
    a["ax"] = a["wax"] * shdg # Eastward
    a["ay"] = a["wax"] * chdg # Northward
    a["az"] = a["waz"]        # Vertical

    # Note, angular velocity and acceleration are in wy direction,
    # Rotate angular velocity from wave to earth
    a["omegax"] =  a["wOmega"] * chdg # Eastward
    a["omegay"] = -a["wOmega"] * shdg # Northward
    a["omegaz"] = np.zeros(a["wOmega"].shape) # Vertical always zero

    # Rotate angular acceleration from wave to earth
    a["alphax"] =  a["wAlpha"] * chdg # Eastward
    a["alphay"] = -a["wAlpha"] * shdg # Northward
    a["alphaz"] = np.zeros(a["wAlpha"].shape) # Vertical always zero

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
