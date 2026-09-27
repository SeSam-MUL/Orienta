"""The same failure, arrived at without anyone writing sys.exit.

ArgumentParser.parse_args() prints usage and exits with code 2 on bad input
rather than raising, so an add-on that parses anything internally ends the
caller's request. This is the version an author hits without realising.
"""
import argparse


def analyse(context):
    parser = argparse.ArgumentParser(prog="demo", add_help=False)
    parser.add_argument("--window", type=int, required=True)
    parser.parse_args([])           # missing required argument -> exit(2)
    return []                       # pragma: no cover - never reached
