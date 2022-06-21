import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import json
import pandas as pd
import multiprocessing
from machine_vision_acquisition_python.calibration.distortion import Undistorter
from machine_vision_acquisition_python.calibration.libcalib import read_calib_parameters


log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--calibio-json",
    "-c",
    "calibio_json_path",
    help="Path to calibio output JSON",
    required=True,
    type=click.types.Path(
        dir_okay=False,
        file_okay=True,
        exists=True,
        readable=True,
        path_type=Path,
        resolve_path=True,
    ),
)
@click.option(
    "--input",
    "-i",
    "input_path",
    help="Input path to read images for undistorting from",
    required=True,
    type=click.types.Path(
        dir_okay=True,
        file_okay=False,
        exists=True,
        readable=True,
        path_type=Path,
        resolve_path=True,
    ),
)
@click.option(
    "--output",
    "-o",
    "output_path",
    help="Output directory to write results to",
    required=False,
    default=None,
    type=click.types.Path(
        dir_okay=True,
        file_okay=False,
        readable=True,
        path_type=Path,
        writable=True,
        resolve_path=True,
    ),
)
def undistort(
    calibio_json_path: Path, input_path: Path, output_path: typing.Optional[Path]
):
    """
    Rectify images using CalibIO and OpenCV.
    """

    # Ensure output exists
    datetime_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_path is None:
        output_path = (input_path / "outputs" / f"{datetime_path}").resolve()
        log.debug(f"Output path defaulted to: {output_path}")
    output_path.mkdir(exist_ok=True, parents=True)

    calibrationts = read_calib_parameters(calibio_json_path)
    # Todo: fix this somehow camera serial mappings
    calibration = calibrationts[0]

    undistorter = Undistorter(calibration)
    for file_path in input_path.rglob("*.png"):
        image = cv2.imread(str(file_path))
        if not undistorter.initialised:
            undistorter.init_optimal_matrix(image.shape)
        undistored = undistorter.undistort(image)
        output_file_path = output_path / f"{file_path.stem}-undistorted.png"
        if not cv2.imwrite(str(output_file_path), undistored):
            raise ValueError(f"Failed to write {output_file_path.name}")
        log.info(f"Undistorted {file_path.name}")
