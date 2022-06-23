"""
Copyright (C) Chronoptics, Ltb - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.
Written by Refael Whyte <r.whyte@chronoptics.com>, 2019

Example viewer for the Chronoptics Kea Camera using python
"""

import chronoptics.tof as tof
from ChronoWidgets import DisplayDevices, DisplayDictStream, Chrono_Blank, get_csf_names
import numpy as np

import tkinter as tk
from tkinter import filedialog


class KeaViewer(tk.Toplevel):
    def __init__(self, master, config_file=None):
        # self.cam = tof.KeaCamera()
        self.master = master

        self.master.wm_title("Chronoptics Viewer")

        self.config_file = config_file

        # The buttons on our window
        self.but_start = tk.Button(
            self.master, text="Stream Camera", command=self.start_camera
        )
        self.but_csf = tk.Button(self.master, text="Stream CSF", command=self.start_csf)
        self.but_stop = tk.Button(self.master, text="Stop", command=self.stop_disp)
        self.but_exit = tk.Button(self.master, text="Quit", command=self.kea_exit)

        self.but_start.grid(row=0, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        tk.Grid.rowconfigure(self.master, 0, weight=1)
        tk.Grid.columnconfigure(self.master, 0, weight=1)

        self.but_csf.grid(row=1, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        tk.Grid.rowconfigure(self.master, 1, weight=1)

        self.but_stop.grid(row=2, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        tk.Grid.rowconfigure(self.master, 2, weight=1)

        self.but_exit.grid(row=3, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        tk.Grid.rowconfigure(self.master, 3, weight=1)

        self.master.protocol("WM_DELETE_WINDOW", self.kea_exit)

        self.is_streaming = False

        return

    def __del__(self):
        return

    def stop_disp(self):
        try:
            if self.cam.isStreaming():
                self.cam.stop()

        except RuntimeError:
            print("Failed to Stop Camera, possibly already stopped")
        except AttributeError:
            pass

        self.is_streaming = False

    def load_json(self):
        file_loc = filedialog.askopenfilename(
            title="Select Filter Config File",
            filetypes=(("Json files", "*.json"), ("all files", "*.*")),
        )

        print(file_loc)

        if file_loc:
            self.config_file = tof.ProcessingConfig(file_loc)
        else:
            self.config_file = tof.ProcessingConfig()

    def start_csf(self):
        if not self.is_streaming:
            if not self.config_file:
                # Get the configuration file
                self.load_json()
                if not self.config_file:
                    print("WARNING: No filter configuration file selected")
                    return

            csf_file = filedialog.askopenfilename(
                title="Select CSF File",
                filetypes=(("CSF files", "*.csf"), ("all files", "*.*")),
            )
            if not csf_file:
                print("No CSF File selected")
                return

            try:
                self.cam = tof.CsfCamera(self.config_file, csf_file)
                self.cam.setFps(20)
                self._start_stream()
            except RuntimeError:
                print("Failed to start stream from CSF file!")

        return

    def _start_stream(self):
        # We have selected the camera, now start streaming.
        stream_list = self.cam.getStreamList()
        # Now display the stream list

        # We get the stream types to display
        streams = DisplayDictStream(self.master, stream_list)
        self.master.wait_window(streams.top)
        sel_streams = streams.get_sel()
        if not sel_streams:
            print("Failed to select display straems")
            self.stop_disp()
            return

        # print(str(sel_streams))
        self.use_streams = []
        for stream in sel_streams:
            self.use_streams.append(stream_list[stream])

        # print(str(self.use_streams))

        # Set the stream list
        self.cam.setStreamList(self.use_streams)

        roi = self.cam.getCameraConfig().getRoi(0)

        self.sz = np.zeros(2, dtype=np.int)
        self.sz[0] = roi.getImgRows()
        self.sz[1] = roi.getImgCols()

        self.disp_list = len(self.use_streams) * [None]
        n = 0
        csf_names = get_csf_names()

        for stream in self.use_streams:
            self.disp_list[n] = Chrono_Blank(
                self.master, self.sz, csf_names[stream.frameType()]
            )
            n += 1

        self.streams = sel_streams

        self.cam.start()
        self.is_streaming = self.cam.isStreaming()
        self.master.after(10, self.update_image)

        return

    def start_camera(self):
        if not self.is_streaming:
            if not self.config_file:
                # Get the configuration file
                self.load_json()

            cam = tof.GigeInterface()
            msgsDict = cam.discover()

            d = DisplayDevices(self.master, msgsDict)
            self.master.wait_window(d.top)
            sel_cam = d.get_sel()
            if not sel_cam:
                print("No Camera selected")
                return
            try:
                self.cam = tof.KeaCamera(
                    self.config_file, msgsDict[int(sel_cam[0])].serial()
                )

                self._start_stream()

            except RuntimeError as err:
                print("start_camera() - Failed: " + str(err))
                self.stop_disp()
                return
        return

    def update_image(self):
        csf_names = get_csf_names()
        if self.is_streaming:
            try:
                frames = self.cam.getFrames()
                clims = {
                    0: [0, 4027],
                    1: [0, 16],
                    2: [-2047, 2047],
                    3: [-1000, 1000],
                    4: [-1000, 1000],
                    5: [0, 6.28],
                    6: [0, 1000],
                    7: [0, 65000],
                    8: [0, 255],
                    9: [-1000, 1000],
                    10: [-1000, 1000],
                    11: [0, 7000],
                }

                n = 0
                for disp in self.disp_list:
                    disp.update_img(
                        np.asarray(frames[n]),
                        clims[frames[n].frameType()],
                        frames[n].frameCount(),
                    )
                    disp.top.wm_title(csf_names[frames[n].frameType()])
                    n = n + 1
                for frame in frames:
                    frame.__del__()
            except RuntimeError:
                if not self.cam.isStreaming():
                    self.stop_disp()
                    print("Failed to update image")
                    return

            self.master.after(10, self.update_image)
        else:
            print("Camera is not streaming")
        return

    def kea_exit(self):
        # Shutdown
        self.stop_disp()
        self.master.destroy()
        self.master.quit()
        return


if __name__ == "__main__":
    print("KeaViewerExample")
    # Run the Moa Viewer

    root = tk.Tk()
    app = KeaViewer(root)
    root.mainloop()
