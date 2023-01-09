from pathlib import Path
import numpy as np
from machine_vision_acquisition_python.calibration.shared import Calibration
from machine_vision_acquisition_python.calibration.interface import (
    load_from_mva_json,
    save_to_mva_json,
)
from machine_vision_acquisition_python.calibration.libcalib import (
    load_from_calibio_json,
)
import pytest

# def test_calibration_load_mva():
#     calib_path = Path(r"/workspaces/sparse-ml-feature-stereo-depth/calibration.json")
#     calibs = load_from_mva_json(calib_path)

#     assert len(calibs)==2

#     tmp_path = calib_path.with_suffix(".tmp.json")
#     save_to_mva_json(calibs, tmp_path)
#     calibs2 = load_from_mva_json(tmp_path)

#     assert len(calibs2)==2

# def test_calibration_load_mva():
#     calib_path = Path(r"/workspaces/sparse-ml-feature-stereo-depth/calibration.json")
#     calibs = load_from_mva_json(calib_path)
#     point_left = [1185, 705]
#     point_right = [1202, 689]
#     from machine_vision_acquisition_python.process.stereo.sparse import SparseStereoProcessor
#     stereo = SparseStereoProcessor(
#        calibs[0],
#        calibs[1]
#     )

#     disp = stereo.disparity_from_dual_points(point_left, point_right)
#     assert round(disp) == 235

#     assert round(stereo.disparity_from_dual_points([1476, 138], [1621, 105])) == 110

#     depth = stereo.disparity_to_depth_mm(disp)
#     assert depth == pytest.approx(730.30885)


def test_sparse_stereo_general():
    """
    Tests
    """
    calib_path = Path(r"test/data/2022-12-22-BioEng-Calib.json")
    calibs = load_from_calibio_json(calib_path)
    p1_left = (779, 851)  # (x,y)
    p2_left = (
        1360,
        824,
    )  # Nose to tail fork, 244mm, unknown dy, back estimated (from output)
    p1_right = (837, 837)
    p2_right = (1419, 809)
    p1_p2_mm = (244, 10)  # (dx,dy)
    from machine_vision_acquisition_python.process.stereo.sparse import (
        SparseStereoProcessor,
    )

    stereo = SparseStereoProcessor(calibs[0], calibs[1])

    disp_px = stereo.disparity_from_dual_points(p1_left, p1_right)
    assert 195.58 == pytest.approx(
        disp_px, 0.01
    )  # Manually confirmed value for this calibration and point

    depth_mm = stereo.disparity_to_depth_mm(disp_px)
    assert 981.51 == pytest.approx(
        depth_mm, 0.01
    )  # Manually confirmed value for this calibration and point

    # get distance between points in mm not accounting for twist (z differences)
    left_points = (
        np.array([p1_left, p2_left]).reshape(-1, 1, 2).astype(np.float32)
    )  # Make a 1xN/Nx1 2-channel CV_32FC2 array
    left_points_undistorted = stereo.undistort_image_points_l(left_points)
    diff_p1_p2 = np.abs(left_points_undistorted[0] - left_points_undistorted[1])[0]
    diff_p1_p2_mm = calibs[0].mm_per_px_at_z(depth_mm) * diff_p1_p2
    assert 242.8 == pytest.approx(
        diff_p1_p2_mm[0], 0.01
    )  # Manually confirmed value for this calibration and point
    err_percentage = (p1_p2_mm - diff_p1_p2_mm)[0] / p1_p2_mm[0] * 100  # Only do dx
    assert err_percentage < 0.5

    # get distance between points in 3d space using point projection
    disp_p2_px = stereo.disparity_from_dual_points(p2_left, p2_right)
    left_points = [
        np.append(left_points_undistorted[0], disp_px),
        np.append(left_points_undistorted[1], disp_p2_px),
    ]  # Make a 1xN/Nx1 3-channel array ready for casting
    left_points_3d = stereo.points_px_to_3d_world_space(left_points)
    diff_3d_p1_p2_mm = left_points_3d[0] - left_points_3d[1]
    pass
