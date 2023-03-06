import click
from pathlib import Path
import typing
import datetime
import cv2
import logging
import json
import multiprocessing
from machine_vision_acquisition_python.process.processing import cvt_tonemap_image

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
@click.option(
    "--out-ext",
    "-e",
    help="File extension. Defaults to input file extension",
    default=None,
)
@click.option(
    "--overwrite",
    help="Whether to overwrite output files. Defaults to false.",
    is_flag=True,
)
@click.option(
    "--skip-existing",
    help="Whether to skip existing output files. Defaults to true.",
    is_flag=True,
)
@click.pass_context
def convert(
    ctx: click.Context,
    input_path: Path,
    output_path: typing.Optional[Path],
    tonemap: bool,
    out_ext: str = None,
    overwrite: bool = False,
    skip_existing: bool = True,
    skip_broken: bool = True,
):
    """
    Batch converts raw 12bit 'PNG' images to de-bayered 12bit images. Optionally tonemaps to 8bit images
    """

    nproc = (
        ctx.parent.params.get("nproc", multiprocessing.cpu_count())
        if ctx.parent
        else multiprocessing.cpu_count()
    )

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
    process_folder(
        input_path, output_path, tonemap, out_ext, overwrite, skip_existing, skip_broken, nproc=nproc
    )


def process_folder(
    input_path: Path,
    output_path: Path,
    tonemap: bool,
    out_ext: str = None,
    overwrite: bool = False,
    skip_existing: bool = True,
    skip_broken: bool = True,
    nproc: int = multiprocessing.cpu_count(),
):
    # delayed import to avoid circular dependency but allow global nproc
    process_args = []
    # Gather work
    for file_path in input_path.rglob("*.png"):
        out_dir = output_path / file_path.parent.relative_to(input_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        process_args.append(
            (file_path.resolve(), out_dir.resolve(), tonemap, out_ext, overwrite, skip_existing, skip_broken)
        )

    # Multiprocess
    pool = multiprocessing.Pool(processes=nproc)
    try:
        log.info("Processing {} files in {}".format(len(process_args), str(input_path)))
        pool.starmap(process_file, process_args)
        log.info(f"Done processing folder {input_path.name}")
    except KeyboardInterrupt as _:
        log.warning("Aborting processing")
    finally:
        pool.close()
        pool.terminate()
        pool.join()


def process_file(
    in_path: Path,
    out_dir: Path,
    tonemap: bool,
    out_ext: str = None,
    overwrite: bool = False,
    skip_existing: bool = True,
    skip_broken: bool = True,
):
    """
    Multoprocessing entrypoint for doing the grunt work
    """
    tonemap_path = ".tonemapped" if tonemap else ""
    out_ext = in_path.suffix if out_ext is None else f".{out_ext.lstrip('.')}"
    out_path = out_dir / f"{in_path.stem}{tonemap_path}{out_ext}"
    if out_path.is_file():
        if skip_existing:
            return
        elif overwrite != True:
            raise FileExistsError(out_path)
    log.debug(f"Processing {in_path} to {out_path}")

    image = cv2.imread(str(in_path), cv2.IMREAD_ANYDEPTH)
    if image is None or not image.any():
        if skip_broken:
            log.warn(f"Unable to read image {in_path}.")
        else:
            raise ValueError(f"Failed to read {in_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    if tonemap:
        image = cvt_tonemap_image(image)
    if not cv2.imwrite(str(out_path), image):
        raise ValueError(f"Failed to write {out_path.name}")
