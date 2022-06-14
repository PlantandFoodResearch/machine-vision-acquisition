import logging
from machine_vision_acquisition_python.capture import capture

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    capture.cli()
