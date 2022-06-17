import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import json
import multiprocessing
from machine_vision_acquisition_python.converter.processing import cvt_tonemap_image
from machine_vision_acquisition_python.utils import get_image_mean, get_image_sharpness, get_image_std


logging.basicConfig(level=logging.DEBUG)
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
@click.option(
    "--tonemap", "-t", help="Output 8bit tonemapped images", is_flag=True, default=False
)
def cli(input_path: Path, output_path: typing.Optional[Path], tonemap: bool):
    """
    Batch converts raw 12bit 'PNG' images to de-bayered 12bit images. Optionally tonemaps to 8bit images
    """

    # Ensure output exists
    if output_path is None:
        datetime_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tonemap_path = "-tonemapped" if tonemap else ""
        output_path = (
            Path.cwd() / "tmp" / f"{input_path.name}-{datetime_path}{tonemap_path}"
        ).resolve()
        log.debug(f"Output path defaulted to: {output_path}")
    output_path.mkdir(exist_ok=True, parents=True)

    # get going
    process_folder(input_path, output_path, tonemap)


def process_folder(input_path: Path, output_path: Path, tonemap: bool):
    process_args = []
    # Gather work
    for file_path in input_path.rglob("*.png"):
        out_dir = output_path / file_path.parent.relative_to(input_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        process_args.append((file_path.resolve(), out_dir.resolve(), tonemap))

    # Multiprocess
    pool = multiprocessing.Pool(processes=8)
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


def process_file(in_path: Path, out_dir: Path, tonemap: bool):
    """
    Multoprocessing entrypoint for doing the grunt work
    """
    tonemap_path = ".tonemapped" if tonemap else ""
    out_path = out_dir / f"{in_path.stem}{tonemap_path}{in_path.suffix}"
    log.debug(f"Processing {in_path} to {out_path}")
    image = cv2.imread(str(in_path), cv2.IMREAD_ANYDEPTH)
    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    if tonemap:
        image = cvt_tonemap_image(image)
    if not cv2.imwrite(str(out_path), image):
        raise ValueError(f"Failed to write {out_path.name}")


def process_folder_stats(input_path: Path, output_path: Path):
    process_args = []
    result_queue: multiprocessing.Queue[typing.Dict] = multiprocessing.Queue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Gather work
    for file_path in input_path.rglob("*.png"):
        process_args.append((file_path.resolve(), result_queue))

    # Multiprocess
    pool = multiprocessing.Pool(processes=8)
    try:
        log.info("Processing {} files in {}".format(len(process_args), str(input_path)))
        pool.starmap(process_file_stats, process_args)
        log.info("Done :)")
        # Gather results
        resuls = gather_results(result_queue)
        output_path.write_text(json.dumps(resuls))
    except KeyboardInterrupt as _:
        log.warning("Aborting processing")
    finally:
        pool.close()
        pool.terminate()
        pool.join()


def process_file_stats(in_path: Path, result_queue: multiprocessing.Queue[typing.Dict] ):
    """
    Multoprocessing entrypoint for doing the grunt work
    """
    log.debug(f"Processing {in_path}")
    image = cv2.imread(str(in_path), cv2.IMREAD_ANYDEPTH)
    sharpness = get_image_sharpness(image)
    mean = get_image_mean(image)
    std = get_image_std(image)
    outputs = {
        "file_name": str(in_path.name),
        "sharpness": sharpness,
        "mean": mean,
        "std": std
    }
    result_queue.put(outputs)


def gather_results(result_queue: multiprocessing.Queue[typing.Dict]):
    results = []
    while not result_queue.empty():
        entry = result_queue.get_nowait()
        results.append(entry)
    return results


if __name__ == "__main__":
    cli()
