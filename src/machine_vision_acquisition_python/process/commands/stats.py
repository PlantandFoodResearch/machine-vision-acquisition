import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import json
import pandas as pd
import multiprocessing
from machine_vision_acquisition_python.process.processing import cvt_tonemap_image
from machine_vision_acquisition_python.utils import get_image_mean, get_image_sharpness, get_image_std, get_image_max


log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    help="Input path to read images for conversion from",
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
def stats(input_path: Path, output_path: typing.Optional[Path]):
    """
    Generate basic numerical stats from folders of images (reccursive) and output xlsx file
    """

    # Ensure output exists
    datetime_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_path is None:
        output_path = (
            input_path / "outputs"
        ).resolve()
        log.debug(f"Output path defaulted to: {output_path}")
    output_path.mkdir(exist_ok=True, parents=True)
    output_file_path = output_path / f"{datetime_path}-stats.xlsx"

    # get going
    process_folder_stats(input_path, output_file_path)


def process_folder_stats(input_path: Path, output_path: Path):
    process_args = []
    # result_queue: multiprocessing.Queue[typing.Dict] = multiprocessing.Queue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Gather work
    for file_path in input_path.rglob("*.png"):
        process_args.append((file_path.resolve(),))

    # Multiprocess
    pool = multiprocessing.Pool(processes=4)
    try:
        log.info("Processing {} files in {}".format(len(process_args), str(input_path)))
        results = pool.starmap(process_file_stats, process_args)
        log.info("Done :)")
        df_results =pd.DataFrame.from_records(results)
        df_results.to_excel(str(output_path), sheet_name="image_stats")
    except KeyboardInterrupt as _:
        log.warning("Aborting processing")
    finally:
        pool.close()
        pool.terminate()
        pool.join()


def process_file_stats(in_path: Path):
    """
    Multoprocessing entrypoint for doing the grunt work
    """
    log.debug(f"Processing {in_path}")
    image = cv2.imread(str(in_path), cv2.IMREAD_ANYDEPTH)
    sharpness = get_image_sharpness(image)
    max = get_image_max(image)
    mean = get_image_mean(image)
    std = get_image_std(image)
    outputs = {
        "file_name": str(in_path.name),
        "folder_name": str(in_path.parent.name),
        "sharpness": sharpness,
        "max": max,
        "mean": mean,
        "std": std
    }
    return outputs
