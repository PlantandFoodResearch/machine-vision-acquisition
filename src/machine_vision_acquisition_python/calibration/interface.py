from pathlib import Path
import logging
import json
from machine_vision_acquisition_python.calibration.shared import Calibration

log = logging.getLogger(__name__)


def load_from_mva_json(json_path: Path):
    """
    Loads calibrations from an internal JSON style format.
    
    This format is basically OpenCV's representation written to a
    structured JSON file that may contain multiple cameras.
    """
    content = json_path.read_text()