import pytest
from pathlib import Path
# Delayed imports of any MVA content to avoid any issues with monkey patching

@pytest.fixture
def calibrations() -> "Calibration":
    from machine_vision_acquisition_python.calibration.libcalib import (
        load_from_calibio_json,
    )
    """return a loaded test calibration"""
    calib_path = Path(r"test/data/2022-12-22-BioEng-Calib.json")
    calibs = load_from_calibio_json(calib_path)
    return calibs

@pytest.fixture
def sparse_stereo(calibrations) -> "SparseStereoProcessor":
    from machine_vision_acquisition_python.calibration.shared import Calibration
    from machine_vision_acquisition_python.process.stereo.sparse import (
        SparseStereoProcessor,
    )
    """
    Return a general sparse stereo object
    """
    stereo = SparseStereoProcessor(calibrations[0], calibrations[1])
    return stereo