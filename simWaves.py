#! /usr/bin/env python3
#
# Simulate waves for a surfaced glider
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import argparse
import MyLogger
import MyProcess

parser = argparse.ArgumentParser(description="Simulate a series of surface waves")
parser.add_argument("yaml", nargs="+", metavar="fn.yml", help="YAML file(s) to simulate")
MyProcess.addArgs(parser)
MyLogger.addArgs(parser)
args = parser.parse_args()

logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

for fn in args.yaml:
    logger.info("Processing %s", fn)
    MyProcess.process(fn, args, logger)
    logger.info("Done processing %s", fn)
