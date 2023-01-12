import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import json
import multiprocessing
from machine_vision_acquisition_python.process.processing import cvt_tonemap_image
from machine_vision_acquisition_python.utils import (
    get_image_mean,
    get_image_sharpness,
    get_image_std,
    get_image_max,
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

from machine_vision_acquisition_python.process.commands.stats import stats
from machine_vision_acquisition_python.process.commands.convert import convert
from machine_vision_acquisition_python.process.commands.undistort import undistort
from machine_vision_acquisition_python.process.commands.calibrate import calib

try:
    from machine_vision_acquisition_python.process.commands.stereo import stereo
except ModuleNotFoundError as _:
    # has optional dependencies, let is fail
    log.warning("mva_process stereo won't be available, imports failed.")
    stereo = None

_commands = [x for x in [stats, convert, undistort, stereo, calib] if x]

_DEFAULT_NPROC = multiprocessing.cpu_count()


# Add each sub-command here :)
@click.group(commands=_commands)
@click.option("--debug", is_flag=True)
@click.option(
    "--nproc",
    default=_DEFAULT_NPROC,
    help="Number of processes to use if multiprocessing",
)
@click.pass_context
def cli(ctx, debug: bool, nproc: int):
    """
    A set of tools to work on folders of images. Check the help for each sub-command for details.
    """
    if debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)
    logging.root.handlers[0].setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )


if __name__ == "__main__":
    cli()
