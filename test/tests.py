from pathlib import Path
from machine_vision_acquisition_python.calibration.shared import Calibration
from machine_vision_acquisition_python.calibration.interface import load_from_mva_json, save_to_mva_json
import pytest

# def test_calibration_load_mva():
#     calib_path = Path(r"/workspaces/sparse-ml-feature-stereo-depth/calibration.json")
#     calibs = load_from_mva_json(calib_path)

#     assert len(calibs)==2

#     tmp_path = calib_path.with_suffix(".tmp.json")
#     save_to_mva_json(calibs, tmp_path)
#     calibs2 = load_from_mva_json(tmp_path)

#     assert len(calibs2)==2

def test_calibration_load_mva():
    calib_path = Path(r"/workspaces/sparse-ml-feature-stereo-depth/calibration.json")
    calibs = load_from_mva_json(calib_path)
    point_left = [1185, 705]
    point_right = [1202, 689]
    from machine_vision_acquisition_python.process.stereo.sparse import SparseStereoProcessor
    stereo = SparseStereoProcessor(
       calibs[0],
       calibs[1] 
    )

    disp = stereo.disparity_from_dual_points(point_left, point_right)
    assert round(disp) == 235

    assert round(stereo.disparity_from_dual_points([1476, 138], [1621, 105])) == 110

    depth = stereo.disparity_to_depth_mm(disp)
    assert depth == pytest.approx(730.30885)
