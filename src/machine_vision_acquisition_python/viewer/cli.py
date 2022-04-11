import click
import logging

log = logging.getLogger(__name__)


# temp
import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import cv2
import ctypes
import numpy as np


@click.command()
@click.option("--name", "-n", help="Camera name to connect to. Defaults to None which will randomly select a camera", default=None)
def cli(name):
    try:
        camera: Aravis.Camera = Aravis.Camera.new(name)
    except Exception as exc:
        log.exception("Could not open camera")
        raise exc
    log.info(f"Opened {camera.get_model_name()}")



if __name__ == "__main__":
    cli()
