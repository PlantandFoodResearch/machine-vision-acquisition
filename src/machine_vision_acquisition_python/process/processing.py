import cv2
import ctypes
import typing
import numpy as np


def buffer_to_numpy_16bit_packed(raw_data_ptr, height: int, width: int):
    # Motivation / Credit: https://stackoverflow.com/a/70525330
    # BUG: The pixel formats in the above example were not working. Used http://softwareservices.flir.com/BFS-U3-200S6/latest/Model/public/ImageFormatControl.html
    raw_data = np.frombuffer(raw_data_ptr, np.uint8)
    # data = np.frombuffer(ptr, dtype=np.uint8)
    raw_data_uint16 = raw_data.astype(np.uint16)
    result = np.zeros(raw_data.size * 2 // 3, np.uint16)

    # This is a bit of a mind bender, but achieves shifting around all the data.
    # For reference: [0::3] gives every 3 entry starting at the 0th entry.
    result[0::2] = ((raw_data_uint16[1::3] & 0x0F)) | (raw_data_uint16[0::3] << 4)
    result[1::2] = ((raw_data_uint16[1::3] & 0xF0) >> 4) | (raw_data_uint16[2::3] << 4)
    image = np.reshape(result, (height, width))
    return image


def buffer_to_numpy_16bit(raw_data_ptr, height: int, width: int):
    ptr = ctypes.cast(raw_data_ptr, ctypes.POINTER(ctypes.c_uint16))
    image = np.ctypeslib.as_array(ptr, (height, width))
    return image


def buffer_to_numpy_8bit(raw_data_ptr, height: int, width: int):
    ptr = ctypes.cast(raw_data_ptr, ctypes.POINTER(ctypes.c_uint8))
    image = np.ctypeslib.as_array(ptr, (height, width))
    return image


def cvt_tonemap_image(image: cv2.Mat) -> cv2.Mat:
    image_f32 = image.astype(np.float32)
    tonemap = cv2.createTonemapReinhard()
    image_f32_tonemap = tonemap.process(image_f32)
    image_uint8 = np.uint8(
        np.clip(image_f32_tonemap * 255, 0, 255)
    )  # clip back to uint8
    return image_uint8  # type: ignore


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
