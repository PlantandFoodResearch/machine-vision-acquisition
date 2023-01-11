import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import re
import numpy as np
import multiprocessing
from torch.multiprocessing import Pool, set_start_method, Lock
from machine_vision_acquisition_python.calibration.stereo import StereoProcessorHSM
from machine_vision_acquisition_python.calibration.libcalib import (
    load_from_calibio_json,
)
from machine_vision_acquisition_python.process.commands.convert import cvt_tonemap_image
from machine_vision_acquisition_python.utils import save_png

try:
    set_start_method("spawn")
except RuntimeError:
    pass

log = logging.getLogger(__name__)


def init_child(lock_):
    global lock
    lock = lock_


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
    "--hsm-model",
    "-hm",
    "hsm_model_path",
    help="Path a trained HSM model (see https://github.com/nznobody/high-res-stereo).",
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
    help="Root input path for images. Must contain sub-folders with matching left & right camera serial numbers",
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
    "--serial-left",
    "-sl",
    "serial_left",
    help="Serial number (or identifier) for left camera. Must match part of folder name in root dir and calibration serial in calibio-json",
    required=True,
    type=click.types.STRING,
)
@click.option(
    "--serial-right",
    "-sr",
    "serial_right",
    help="Serial number (or identifier) for right camera. Must match part of folder name in root dir and calibration serial in calibio-json",
    required=True,
    type=click.types.STRING,
)
@click.option(
    "--disparity-max",
    "-dmmax",
    "disparity_max",
    help="Maximum disparity value to use for stereo engine.",
    type=click.types.INT,
    default=1024,
)
@click.option(
    "--disparity-min",
    "-dmin",
    "disparity_min",
    help="Maximum disparity value to use for stereo engine.",
    type=click.types.INT,
    default=340,
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
    "--disparity-16b-normalised",
    "-d16b",
    "disparity_16b_normalised",
    help="Output normalised 16b disparity (PNG)",
    is_flag=True,
    default=True,
)
@click.option(
    "--pointcloud",
    "-p",
    help="Output xyz pointcloud (PLY)",
    is_flag=True,
    default=False,
)
@click.pass_context
def stereo(
    ctx: click.Context,
    calibio_json_path: Path,
    hsm_model_path: Path,
    input_path: Path,
    serial_left: str,
    serial_right: str,
    disparity_max: int,
    disparity_min: int,
    output_path: typing.Optional[Path],
    disparity_16b_normalised: bool,
    pointcloud: bool,
):
    """
    process stereo scan images and output dispparity maps and/or pointclouds
    """

    nproc = (
        ctx.parent.params.get("nproc", multiprocessing.cpu_count())
        if ctx.parent
        else multiprocessing.cpu_count()
    )
    log.warning("This code is experimental!")

    # Ensure output exists
    datetime_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_path is None:
        output_path = (input_path / "outputs" / f"{datetime_path}").resolve()
        log.debug(f"Output path defaulted to: {output_path}")
    output_path.mkdir(exist_ok=True, parents=True)

    calibrations = load_from_calibio_json(calibio_json_path)
    # Try match calibrations to requested serials
    calib_left = None
    calib_right = None
    for calibration in calibrations:
        if calibration.serial in serial_left or serial_left in calibration.serial:
            log.debug(f"Found calibration for left camera {calibration.serial}")
            calib_left = calibration
        elif calibration.serial in serial_right or serial_right in calibration.serial:
            log.debug(f"Found calibration for right camera {calibration.serial}")
            calib_right = calibration
        if calib_left and calib_right:
            break
    else:
        raise ValueError(
            f"Could not match calibration serials ({[calibration.serial for calibration in calibrations]}) to input serials: {[serial_left, serial_right]}"
        )

    # Try match folders to requested serials
    input_dir_left = None
    input_dir_right = None
    for top_level_path in input_path.glob("**/"):
        if serial_left in str(top_level_path) and top_level_path.is_dir():
            log.debug(f"Found left camera folder {top_level_path}")
            input_dir_left = top_level_path
        elif serial_right in str(top_level_path) and top_level_path.is_dir():
            log.debug(f"Found left camera folder {top_level_path}")
            input_dir_right = top_level_path
        if input_dir_left and input_dir_right:
            break
    else:
        raise ValueError(
            f"Could not match camera serials {[serial_left, serial_right]} to input folders in: {input_path}"
        )

    process_args = []
    stereo = StereoProcessorHSM(
        calibration_left=calib_left,
        calibration_right=calib_right,
        min_disparity=disparity_min,
        max_disparity=disparity_max,
        rescale_factor=1.0,
        hsm_model_path=hsm_model_path,
    )

    # Multiproc init
    cuda_lock = Lock()
    # Todo use nproc
    pool = Pool(processes=nproc, initializer=init_child, initargs=(cuda_lock,))

    indexes = []
    for left_file_path in input_dir_left.rglob("*.png"):
        out_file_path = (output_path / left_file_path.name).resolve()
        re_match = re.search(
            r"img(\d*?)_(.*?)_(.*?)_(.*?).png$", str(left_file_path.name)
        )
        if re_match is None or re_match.group(1) is None:
            log.warning(f"Skipping {left_file_path}, could not find image ID")
            continue
        index = int(re_match.group(1), base=10)
        if index in indexes:
            raise IndexError(
                f"Image with index {index} already encountered! This can be resolved, but isn't yet implemented"
            )
        indexes.append(index)
        # match on entire string time values
        # Todo: better would be to confirm xy from JSON files
        right_file_glob = (
            f"img{re_match.group(1)}_{re_match.group(2)}_{re_match.group(3)}_*.png"
        )
        right_files = list(input_dir_right.rglob(right_file_glob))
        if len(right_files) != 1:
            raise FileNotFoundError(
                f"Could not find matching right image for left {left_file_path}"
            )
        right_file = right_files[0].resolve()
        process_args.append(
            (left_file_path.resolve(), right_file, out_file_path, stereo)
        )

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


def process_file(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    stereo: StereoProcessorHSM,
    tonemap: bool = True,
):
    """
    Stereo process single file set for multiprocessing

    Expects raw bayer 12b images
    """
    image_left = cv2.imread(str(left_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    image_right = cv2.imread(str(right_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if (
        image_left is None
        or not image_left.any()
        or image_right is None
        or not image_right.any()
    ):
        raise ValueError(
            f"Could not read left ({left_path}) & right ({right_path}) images."
        )

    # Prepare image
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BayerRG2RGB)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BayerRG2RGB)
    if tonemap:
        image_left = cvt_tonemap_image(image_left)
        image_right = cvt_tonemap_image(image_right)

    image_left, image_right = stereo.remap(image_left, image_right)
    with lock:
        disp = stereo.calculate_disparity(image_left, image_right)

    # TEMP
    # Create fake-disp for viewing
    disparity_shifted = stereo.shift_disp_down(disp)
    disp_visual = stereo.normalise_disparity_8b(disparity_shifted)
    # move to align roughly with original image.
    # NOTE: the value of ~340 is experimentally determined. A proper remap inversion should be done
    # Alternative: -291
    # Possibly: https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid
    tx = -340  # px
    ty = 0  # px
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])  # type: ignore
    num_rows, num_cols = disp_visual.shape[:2]
    disp_visual = cv2.warpAffine(disp_visual, translation_matrix, (num_cols, num_rows))

    if not cv2.imwrite(str(out_path), disp_visual):
        raise IOError(f"Could not save to {out_path}")

    log.debug(f"stereo processed {left_path.name}")
