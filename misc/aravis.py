import gi

gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import cv2
import ctypes
import numpy as np

camera = Aravis.Camera.new(None)
camera.set_frame_rate(1.0)
camera.set_pixel_format(Aravis.PIXEL_FORMAT_BAYER_RG_12)

payload = camera.get_payload()

[x, y, width, height] = camera.get_region()

print("Camera vendor : %s" % (camera.get_vendor_name()))
print("Camera model  : %s" % (camera.get_model_name()))
print("ROI           : %dx%d at %d,%d" % (width, height, x, y))
print("Payload       : %d" % (payload))
print("Pixel format  : %s" % (camera.get_pixel_format_as_string()))

stream = camera.create_stream(None, None)

for i in range(0, 10):
    stream.push_buffer(Aravis.Buffer.new_allocate(payload))

print("Start acquisition")

camera.start_acquisition()

print("Acquisition")

for i in range(0, 20):
    image = stream.pop_buffer()
    print(image)
    if image:
        stream.push_buffer(image)

print("Stop acquisition")

camera.stop_acquisition()
