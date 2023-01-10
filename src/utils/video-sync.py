from pathlib import Path
import re
import subprocess
import logging
import click

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("video-sync")

@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    help="Video file to offset",
    type=click.types.Path(
        dir_okay=False,
        file_okay=True,
        exists=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--time",
    "-t",
    "time_str",
    help="Exact start time in HH:MM:SS.sss format.",
    type=click.types.STRING,
    required=True,
)
def main(
    input_path: Path,
    time_str: str,
):
    """
    Output re-encoded MP4 from input with exact start time. A thin wrapper around ffmpeg.

    E.g. video-sync -i left.mp4 -t 00:00:05.500.

    *Note*: To achieve exact offsets the video must be re-encoded. This is done using libx264
    """
    time_matches = re.search(r"^(\d\d):(\d\d):(\d\d)\.(\d{0,3})$", time_str)
    if time_matches is None:
        raise ValueError(f"Could not extract time from {time_str}, please check formatting")

    output_path = input_path.with_name(f"{input_path.stem}_T{time_matches[1]}-{time_matches[2]}-{time_matches[3]}.{time_matches[4]}{input_path.suffix}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "fatal",
        "-y",  # Overwrite output without asking
        "-an",  # Drop audio
        "-ss",
        f"{time_str}",
        "-accurate_seek",  # Needs to be re-encoded, but won't jump to nearest keyframe
        "-i",
        f"{input_path}",
        "-c:v",
        "libx264",
        "-map_metadata",  # Copy over creation datetime metadata
        "0",
        f"{output_path}",
    ]
    log.info(f"Running command: {' '.join(command)}")
    subprocess.check_call(command)

if __name__ == "__main__":
    main()