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
from machine_vision_acquisition_python.calibration.libcalib import load_from_calibio_json

log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--calibio-json",
    "-c",
    "calibio_json_path",
    help="Path to calibio output JSON. Must contain camera serials to match input path to.",
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
    help="Input path to read images for undistorting from. Must only contain images from a single camera.",
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
@click.option(
    "--force",
    "-f",
    help="Use any calibration, even if cannot match serial",
    is_flag=True,
    default=False,
)
@click.pass_context
def undistort(
    ctx: click.Context,
    calibio_json_path: Path,
    input_path: Path,
    output_path: typing.Optional[Path],
    force: bool,
):
    """
    Rectify images using CalibIO and OpenCV from a single camera.
    """

    nproc = ctx.parent.params.get("nproc", multiprocessing.cpu_count()) if ctx.parent else multiprocessing.cpu_count()

    # Ensure output exists
    datetime_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_path is None:
        output_path = (input_path / "outputs" / f"{datetime_path}").resolve()
        log.debug(f"Output path defaulted to: {output_path}")
    output_path.mkdir(exist_ok=True, parents=True)

    calibrations = load_from_calibio_json(calibio_json_path)
    # Try match serial to folder path
    for calibration in calibrations:
        if calibration.serial in str(input_path.resolve()):
            log.debug(
                f"matched path component to calibration camera serial {calibration.serial}"
            )
            break
    else:
        if not force:
            raise ValueError(
                f"Could not match calibration serials ({[calibration.serial for calibration in calibrations]}) to part of folder path {input_path}"
            )
        log.warning(
            f"Could not match calibration serials to folder path, using first calibration (--force specified)"
        )
        calibration = calibrations[0]
    # Todo: fix this somehow camera serial mappings

    process_args = []
    undistorter = Undistorter(calibration)
    for file_path in input_path.rglob("*.png"):
        if not undistorter.initialised:
            image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
            if image is None or not image.any():
                raise ValueError(f"Failed to read {file_path}")
            undistorter.init_optimal_matrix(image.shape)
        out_dir = (output_path / file_path.parent.relative_to(input_path)).resolve()
        out_dir.mkdir(exist_ok=True, parents=True)
        process_args.append((file_path.resolve(), out_dir, undistorter))

    pool = multiprocessing.Pool(processes=nproc)
    try:
        log.info("Processing {} files in {}".format(len(process_args), str(input_path)))
        pool.starmap(process_file, process_args)
        log.info("Done :)")
    except KeyboardInterrupt as _:
        log.warning("Aborting processing")
    finally:
        pool.close()
        pool.terminate()
        pool.join()


def process_file(in_path: Path, out_dir: Path, undistorter: Undistorter):
    """Undistort single file for multiprocessing"""
    image = cv2.imread(str(in_path), cv2.IMREAD_ANYDEPTH)
    if image is None or not image.any():
        raise ValueError(f"Failed to read {in_path}")
    if not undistorter.initialised:
        undistorter.init_optimal_matrix(image.shape)
    undistored = undistorter.undistort(image)
    output_file_path = out_dir / f"{in_path.stem}.undistorted{in_path.suffix}"
    if not cv2.imwrite(str(output_file_path), undistored):
        raise ValueError(f"Failed to write {output_file_path.name}")
    log.info(f"Undistorted {in_path.name}")
