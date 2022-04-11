import click
import logging
import cv2
import typing
from timeit import default_timer as timer

log = logging.getLogger(__name__)


# temp
import gi

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import cv2
import ctypes
import numpy as np


def convert(buf):
    ### Credit: https://github.com/SintefRaufossManufacturing/python-aravis/blob/master/aravis.py#L181
    if not buf:
        return None
    pixel_format = buf.get_image_pixel_format()
    bits_per_pixel = pixel_format >> 16 & 0xFF
    if bits_per_pixel == 8:
        INTP = ctypes.POINTER(ctypes.c_uint8)
    else:
        INTP = ctypes.POINTER(ctypes.c_uint16)
    addr = buf.get_data()
    ptr = ctypes.cast(addr, INTP)
    im = np.ctypeslib.as_array(ptr, (buf.get_image_height(), buf.get_image_width()))
    im = im.copy()
    return im


def resize_with_aspect_ratio(
    image,
    width: "typing.Optional[int]" = None,
    height: "typing.Optional[int]" = None,
    inter=cv2.INTER_AREA,
):
    # borrowed from https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
    # And logc improved
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        # return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        return image
    if width is None and height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif width is not None and height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        raise ValueError("Cannot specify width and height")
    # cv2.resize(image, dim, interpolation=inter)
    # return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    return cv2.resize(image, dim, interpolation=inter)


@click.command()
@click.option(
    "--name",
    "-n",
    help="Camera name to connect to. Defaults to None which will randomly select a camera",
    default=None,
)
def cli(name):
    try:
        camera: Aravis.Camera = Aravis.Camera.new(name)
    except Exception as exc:
        log.exception("Could not open camera")
        raise exc
    log.info(f"Opened {camera.get_model_name()} {camera.get_device_serial_number()}")
    stream = camera.create_stream(None, None)
    payload = camera.get_payload()

    # Ensure we have frames to buffer
    for i in range(0, 50):
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    log.info("starting acquisition")
    camera.start_acquisition()
    try:
        while True:
            start = timer()
            buffer = stream.try_pop_buffer()
            if buffer:
                image = convert(buffer)
                stream.push_buffer(buffer)  # push buffer back into stream
                image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
                image = resize_with_aspect_ratio(image, width=640)
                cv2.imshow("frame", image)
                ch = cv2.waitKey(10) & 0xFF
                if ch == 27 or ch == ord("q"):
                    break
                elif ch == ord("s"):
                    cv2.imwrite("imagename.png", image)
                end = timer()
                log.debug(f"Loop execution time: {end - start}")
    finally:
        camera.stop_acquisition()


if __name__ == "__main__":
    cli()
