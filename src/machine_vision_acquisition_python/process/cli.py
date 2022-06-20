import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import json
import multiprocessing
from machine_vision_acquisition_python.process.processing import cvt_tonemap_image
from machine_vision_acquisition_python.utils import get_image_mean, get_image_sharpness, get_image_std, get_image_max
from machine_vision_acquisition_python.process.commands.stats import stats
from machine_vision_acquisition_python.process.commands.convert import convert

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


# Add each sub-command here :)
@click.group(commands=[
    stats,
    convert,
])
def cli():
    pass


if __name__ == "__main__":
    cli()
