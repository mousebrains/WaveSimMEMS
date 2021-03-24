#! /usr/bin/env python3
#
# Simulate waves for a surfaced glider
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import argparse
import MyLogger
import MyProcess
import MyPolar

parser = argparse.ArgumentParser(description="Simulate a series of surface waves")
parser.add_argument("yaml", nargs="+", metavar="fn.yml", help="YAML file(s) to simulate")
parser.add_argument("--polar", action="store_true", help="Plot directional spectrum of results")
parser.add_argument("--threshold", type=float, default=0.01,
        help="Polar plot colorbar threshold")
MyProcess.addArgs(parser)
MyLogger.addArgs(parser)
args = parser.parse_args()

logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

for fn in args.yaml:
    logger.info("Processing %s", fn)
    info = MyProcess.process(fn, args, logger)
    logger.info("Done processing %s", fn)

    if args.polar and (info is not None):
        MyPolar.plotit(info, threshold=args.threshold)
