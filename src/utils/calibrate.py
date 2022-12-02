from pathlib import Path
import numpy as np
import re
import concurrent.futures
from multiprocessing import cpu_count
from typing import Tuple, List, Optional
import cv2
import json
import logging
import click

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("calibrate")

_IMG_EXTENSIONS = [".png", ".jpg"]
_BOARD_TYPES = ["CheckerboardMarker"]
_DEBUG = False  # Outputs images with found corners rendered


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    help="Directory of images to use",
    type=click.types.Path(
        dir_okay=True,
        file_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--board-type",
    "-b",
    "board_type",
    help="Calibration board type",
    type=click.types.Choice(_BOARD_TYPES, case_sensitive=False),
    required=True,
)
@click.option(
    "--rows",
    "-r",
    "board_rows",
    help="Number of inner corners per a chessboard column (careful of how to count)",
    type=click.types.INT,
    required=True,
)
@click.option(
    "--columns",
    "-c",
    "board_columns",
    help="Number of inner corners per a chessboard row (careful of how to count)",
    type=click.types.INT,
    required=True,
)
@click.option(
    "--size",
    "-s",
    "board_checker_size",
    help="Size of each checker in mm",
    type=click.types.FLOAT,
    required=True,
)
@click.option(
    "--input-stereo",
    "-is",
    "input_stereo_path",
    help="Second directory of images to use (implies stereo calibration)",
    type=click.types.Path(
        dir_okay=True,
        file_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=False,
)
@click.option(
    "--single-thread",
    "-st",
    "singlethread",
    is_flag=True,
    help="Run calibration in single thread only",
    type=click.types.BOOL,
)
@click.option(
    "--stereo-index-regex",
    "-sr",
    "index_regex",
    help="Regex to extract image index from name for stereo image matching",
    type=click.types.STRING,
    default=r"^(\d*?)\D"  # All leading digits until first non-digit
)
def calib(
    input_path: Path,
    board_type: str,
    board_rows: int,
    board_columns: int,
    board_checker_size: float,
    input_stereo_path: Optional[Path],
    singlethread: bool,
    index_regex: str
):
    """Just another OpenCV Calibration CLI"""
    if singlethread:
        threads = 1
    else:
        threads = cpu_count()
    pattern_size = (board_columns, board_rows)
    single_object_points = generate_checker_board_points(
        board_rows, board_columns, board_checker_size
    )
    expected_shape = None
    rot = None
    trans = None

    # Gather main / left camera
    image_paths, expected_shape = get_image_paths_and_size(input_path)
    image_points, obj_points, image_names = process_images(
        image_paths, single_object_points, expected_shape, pattern_size, threads
    )
    log.info(f"Calibrating {input_path.name} with {len(image_points)} images")

    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, image_points, expected_shape, None, None
    )

    calibration_objs = []
    calibration_objs.append(
        {
            "name": input_path.name,
            "image_size": {"width": expected_shape[1] ,"height": expected_shape[0]},
            "camera_matrix": camera_matrix,
            "dist_coefs": dist_coefs,
            "r_vec": [0,0,0],
            "t_vec": [0,0,0],
            "_rms": rms,
        }
    )

    # Optional stereo
    if input_stereo_path:
        log.info(f"Calibration stereo camera with second path {input_stereo_path}")
        image_paths_stereo, expected_shape_stereo = get_image_paths_and_size(
            input_stereo_path
        )
        if expected_shape != expected_shape_stereo:
            raise ValueError("Stereo camera shape must match")
        image_points_stereo, obj_points_stereo, image_names_stereo = process_images(
            image_paths_stereo,
            single_object_points,
            expected_shape_stereo,
            pattern_size,
            threads,
        )

        if len(image_points_stereo) != len(image_points):
            ValueError("Must have equal image count")

        (
            rms_stereo,
            camera_matrix_stereo,
            dist_coefs_stereo,
            _rvecs_stereo,
            _tvecs_stereo,
        ) = cv2.calibrateCamera(
            obj_points_stereo, image_points_stereo, expected_shape_stereo, None, None
        )

        # Align image_points
        image_points_aligned, image_points_stereo_aligned =  align_image_points(image_points, image_names, image_points_stereo, image_names_stereo, index_regex)

        # Perform actual stereo calibration
        (
            retStereo,
            camera_matrix_L,
            dist_coefs_L,
            camera_matrix_R,
            dist_coefs_R,
            rot,
            trans,
            essentialMatrix,
            fundamentalMatrix,
        ) = cv2.stereoCalibrate(
            obj_points,
            image_points_aligned,
            image_points_stereo_aligned,
            camera_matrix,
            dist_coefs,
            camera_matrix_stereo,
            dist_coefs_stereo,
            expected_shape,
        )
        calibration_objs.append(
            {
                "name": input_stereo_path.name,
                "image_size": {"width": expected_shape[1] ,"height": expected_shape[0]},
                "camera_matrix": camera_matrix_stereo,
                "dist_coefs": dist_coefs_stereo,
                "r_vec": rot,
                "t_vec": trans,
                "_rms": rms_stereo,
            }
        )

    camera_matrix_path = Path.cwd() / "calibration.json"
    camera_matrix_str = json.dumps(calibration_objs, cls=NumpyEncoder)
    camera_matrix_path.write_text(camera_matrix_str)

    # Attempt to output opencv YAML format as well
    cv2_calib_path = Path.cwd() / "calibration.yml"
    cv_file = cv2.FileStorage(str(cv2_calib_path), cv2.FILE_STORAGE_WRITE)
    try:
        cv_file.write('M1', camera_matrix)
        cv_file.write('D1', dist_coefs)
        cv_file.write('M2', camera_matrix_stereo)
        cv_file.write('D2', dist_coefs_stereo)
        if rot is not None:
            cv_file.write('R', rot)
            cv_file.write('T', trans)
    finally:
        cv_file.release()


def align_image_points(image_points:List , image_names: List[str], image_points_stereo: List, image_names_stereo: List[str], index_regex:str):
    """Finds indicies based on image name list (must match order of points list) and re-orders both lists to match"""
    out_image_points = []
    out_image_points_stereo = []
    _tmp_dict = {}
    _tmp_dict_stereo = {}
    for image_point, image_name in zip(image_points_stereo, image_names_stereo):
        image_index = get_image_index(image_name, index_regex)
        _tmp_dict_stereo.update({image_index: image_point})
    # Not strictly necessary, but let's us sort them both
    for image_point, image_name in zip(image_points, image_names):
        image_index = get_image_index(image_name, index_regex)
        _tmp_dict.update({image_index: image_point})
    for index in sorted(_tmp_dict.keys()):
        if index not in _tmp_dict_stereo:
            raise ValueError("Could not find matching indexes in image names")
        out_image_points.append(_tmp_dict.get(index))
        out_image_points_stereo.append(_tmp_dict_stereo.get(index))
    return out_image_points, out_image_points_stereo



def get_image_index(image_name: str, index_regex: str):
    matches = re.search(index_regex, image_name)
    if matches is not None:
        return int(matches[1])
    else:
        raise ValueError(f"Could not extract index from {image_name}")



def get_image_paths_and_size(base_path) -> Tuple[List[Path], Tuple[int, int]]:
    """Return a list of paths of valid images and also expected size (from samplling first image)"""
    image_paths = []
    expected_shape = None
    for image_path in base_path.glob("*"):
        if image_path.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        if not expected_shape:
            # Get expected image size from first image
            img = cv2.imread(
                str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
            )  # TODO: use imquery call to retrieve results
            expected_shape = img.shape[:2]
            del img
        image_paths.append(image_path)
    return image_paths, expected_shape


def process_images(
    image_paths: List[Path], single_obj_points, expected_shape, pattern_size, threads=1
):
    """Multiprocess list of images and return image and object point lists"""
    futures = []
    image_points = []
    obj_points = []
    image_names = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for image_path in image_paths:
            futures.append(
                executor.submit(
                    process_single_image, image_path, expected_shape, pattern_size
                )
            )

        # Will this preserve order?
        done, not_done = concurrent.futures.wait(futures)
        # Gather points
        for future in done:
            try:
                result = future.result()
            except Exception as exc:
                log.warning("Skipping expected future")
            else:
                if result is not None:
                    points, image_name = result
                    image_points.append(points)
                    obj_points.append(single_obj_points)
                    image_names.append(image_name)
    if not image_points:
        raise ValueError(f"Could not find any valid calibration targets")
    return image_points, obj_points, image_names


def generate_checker_board_points(rows: int, columns: int, square_size: float):
    pattern_size = (columns, rows)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    return pattern_points


def process_single_image(
    img_path: Path,
    expected_shape: Tuple[int, int],
    pattern_size: Tuple[int, int],
):
    # log.debug(f"processing {img_path}... ")
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        log.warning(f"Failed to load {img_path}, skipping")
        return
    if len(img.shape) == 3:
        # Convert to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if expected_shape != img.shape[:2]:
        log.warning(f"{img_path} is not expected size, skipping")
        return
    found, corners = cv2.findChessboardCornersSB(
        img, pattern_size, flags=cv2.CALIB_CB_MARKER | cv2.CALIB_CB_ACCURACY
    )
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if _DEBUG:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        outfile = (
            img_path.parent.parent / "DEBUG" / img_path.parent.name / img_path.name
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outfile), vis)

    if not found:
        log.warning("chessboard not found")
        return None

    log.debug(f"           {img_path}... OK")
    return corners.reshape(-1, 2), img_path.name


if __name__ == "__main__":
    calib()
