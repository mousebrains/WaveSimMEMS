#! /usr/bin/env python3

import yaml
import logging

def load(fn:str, logger:logging.Logger = None) -> dict:
    try:
        with open(fn, "r") as fp:
            lines = fp.read()
            if logger is not None:
                logger.info("Processing %s\n%s", fn, lines)
            data = yaml.load(lines, Loader=yaml.SafeLoader)
            return data
    except Exception as e:
        if logger is not None:
            logger.exception("Error loading %s", fn)
        raise e

if __name__ == "__main__":
    import argparse
    import MyLogger

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="+", metavar="fn.yml", help="YAML file(s) to load")
    MyLogger.addArgs(parser)
    args = parser.parse_args()

    logger = MyLogger.mkLogger(args, __name__, "%(asctime)s: %(levelname)s - %(message)s")

    for fn in args.yaml:
        data = load(fn, logger)
        logger.info("fn %s\n%s", fn, data)

