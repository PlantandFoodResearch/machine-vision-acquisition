import logging
from machine_vision_acquisition_python.capture import capture

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    capture.cli()
