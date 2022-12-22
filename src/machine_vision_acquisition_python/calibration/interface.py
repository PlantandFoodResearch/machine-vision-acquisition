from pathlib import Path
import logging
import json
from typing import List
import numpy as np
from machine_vision_acquisition_python.calibration.shared import Calibration, CameraModel

log = logging.getLogger(__name__)


def load_from_mva_json(json_path: Path) -> List[Calibration]:
    """
    Loads calibrations from an internal JSON style format.
    
    This format is basically OpenCV's representation written to a
    structured JSON file that may contain multiple cameras.
    """
    content = json.loads(json_path.read_text())
    calibs = []
    for calibration in content:
        name = calibration["name"]
        width = calibration["image_size"]["width"]
        height = calibration["image_size"]["height"]
        camera_matrix = calibration["camera_matrix"]
        dist_coefs = calibration["dist_coefs"]
        rotation = calibration["r_vec"]
        translation = calibration["t_vec"]
        camera_model = calibration["camera_model"]

        # marshal types
        camera_matrix = np.matrix(camera_matrix, dtype=np.float64)
        dist_coefs = np.array(dist_coefs, dtype=np.float64)
        rotation = np.array(rotation, dtype=np.float64)
        translation = np.array(translation, dtype=np.float64)
        width = int(width)
        height = int(height)
        camera_model = CameraModel(camera_model)

        # Always store rotation in rotation vector notation (rx,ry,rz)
        if len(rotation.shape)!=1 and rotation.shape == (3,3):
            log.debug(f"Converting matrix to rotation vector")
            import cv2
            rotation, _ = cv2.Rodrigues(rotation)
            # opencv gives back (3,1) array instead of just (3,)
            rotation = rotation.reshape((3,))
        
        calib_obj = Calibration(
            name_or_serial=name,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coefs,
            rvec=rotation,
            tvec=translation,
            image_height=height,
            image_width=width,
            camera_model=camera_model
        )
        calibs.append(calib_obj)
    return calibs


def save_to_mva_json(calibs: List[Calibration], out_path: Path):
    """
    Saves list of Calibration objects to MVA compatible format (basically OpenCV).
    """
    output_objs = []
    for calib in calibs:
        dict_calib = {
            "name": calib.serial,
            "camera_model": str(calib.camera_model),
            "image_size" : {
                "width": calib.image_width,
                "height": calib.image_height
            },
            "camera_matrix": calib.cameraMatrix.tolist(),
            "dist_coefs": calib.distCoeffs.tolist(),
            "r_vec": calib.rvec.tolist(),
            "t_vec": calib.tvec.tolist(),
        }
        output_objs.append(dict_calib)
    out_path.write_text(json.dumps(output_objs))
