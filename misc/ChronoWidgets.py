"""
Copyright (C) Chronoptics, Ltb - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.  
Written by Refael Whyte <r.whyte@chronoptics.com>, 2019

Tkinter Widgets for displaying images. 
"""
import sys
from typing import List

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import cm
from PIL import Image
from PIL import ImageTk

import tkinter as tk
import scipy.misc
import numpy as np
import chronoptics.tof as tof


def get_csf_names():
    return {v: k for k, v in vars(tof.FrameType).items() if not k.startswith("__")}


class DisplayDictStream(object):
    def __init__(self, master, stream_list: List[tof.Stream]):
        top = self.top = tk.Toplevel(master)
        top.wm_title("Data Streams")

        # Make the GUI resizeable
        tk.Grid.columnconfigure(self.top, 0, weight=1)
        tk.Grid.columnconfigure(self.top, 1, weight=1)

        # tk.Grid.rowconfigure(self.top,0, weight=1)
        tk.Grid.rowconfigure(self.top, 1, weight=1)
        # We might not want to scale this proportionaly
        # tk.Grid.rowconfigure(self.top,2, weight=1)

        self.stream_label = tk.Label(top, text="Avaliable Data Streams")
        self.Lb1 = tk.Listbox(top, selectmode=tk.EXTENDED)

        csf_names = get_csf_names()

        for stream in stream_list:
            self.Lb1.insert(tk.END, str(csf_names[stream.frameType()]))

        self.stream_label.grid(
            row=0, column=0, columnspan=2, sticky=tk.S + tk.N + tk.W + tk.E
        )
        self.Lb1.grid(row=1, column=0, columnspan=2, sticky=tk.S + tk.N + tk.W + tk.E)

        self.but_okay = tk.Button(top, text="Okay", command=self.ok)
        self.but_cancel = tk.Button(top, text="Cancel", command=self.cancel)

        self.but_okay.grid(row=2, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.but_cancel.grid(row=2, column=1, sticky=tk.S + tk.N + tk.W + tk.E)

    def ok(self):
        self.sel_item = self.Lb1.curselection()
        self.top.destroy()
        return

    def cancel(self):
        self.sel_item = []
        self.top.destroy()
        return

    def get_sel(self):
        return self.sel_item


class DisplayDevices(object):
    def __init__(self, master, msgsDict):

        top = self.top = tk.Toplevel(master)
        top.wm_title("Devices")

        self.sel_item = []
        self.cam_label = tk.Label(top, text="Avaliable Devices")

        self.Lb1 = tk.Listbox(top, selectmode=tk.SINGLE)
        n = 0
        for msg in msgsDict:
            self.Lb1.insert(n, str(msg.serial()))
            n += 1

        self.cam_label.grid(
            row=0, column=0, columnspan=2, sticky=tk.S + tk.N + tk.W + tk.E
        )
        self.Lb1.grid(row=1, column=0, columnspan=2, sticky=tk.S + tk.N + tk.W + tk.E)

        self.but_okay = tk.Button(top, text="Okay", command=self.ok)
        self.but_cancel = tk.Button(top, text="Cancel", command=self.cancel)

        self.but_okay.grid(row=2, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.but_cancel.grid(row=2, column=1, sticky=tk.S + tk.N + tk.W + tk.E)

        # tk.Grid.rowconfigure(self.top,0, weight=1)
        tk.Grid.rowconfigure(self.top, 1, weight=1)
        # tk.Grid.rowconfigure(self.top,2, weight=1)

        tk.Grid.columnconfigure(self.top, 0, weight=1)
        tk.Grid.columnconfigure(self.top, 1, weight=1)

    def ok(self):
        # Get the currently selected ID
        self.sel_item = self.Lb1.curselection()
        # print('sel_items: ' + str(self.sel_item))
        self.top.destroy()

    def cancel(self):
        self.sel_item = []
        self.top.destroy()

    def get_sel(self):
        return self.sel_item


class Chrono_Canvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        tk.Canvas.__init__(self, master, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def cal_scale(self, img):
        # Calculate how much we have to scale the image by to fit on the canvas
        row = np.size(img, axis=0)
        col = np.size(img, axis=1)
        # Now figure out how much to resize my
        row_scale = float(self.height) / float(row)
        col_scale = float(self.width) / float(col)
        # We take the smallest scale factor as it is the one that can fit
        scale = np.min(np.array([row_scale, col_scale]))
        return scale

    def add_image(self, img):
        # Add an image to the canvas
        self.img = img
        self.scale_factor = self.cal_scale(img)
        # img_sc = scipy.misc.imresize(img, self.scale_factor, interp='nearest')
        img_sc = np.uint8(img)
        img_can = Image.fromarray(img_sc)
        self.imgTk = ImageTk.PhotoImage(img_can)
        self.image_on_canvas = self.create_image(
            int(self.width / 2), int(self.height / 2), image=self.imgTk
        )
        return

    def update_image(self, img):
        # Update the image on the canvas
        # To convert to a jet color map
        # self.img = cm.jet( img )
        # print(str(img.dtype))
        # print(str(np.shape(img)))
        self.img = img
        self.scale_factor = self.cal_scale(self.img)
        # img_sc = scipy.misc.imresize(self.img, self.scale_factor, interp='nearest')
        img_sc = np.uint8(img * 255.0)
        img_can = Image.fromarray(img_sc)
        self.imgTk = ImageTk.PhotoImage(img_can)
        self.itemconfig(self.image_on_canvas, image=self.imgTk)
        return

    def on_resize(self, event):
        # print("on_resize: " + str(event.width) + " " + str(event.height) )
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        # print("on_resize: wscale,hscale " + str(wscale) + "," + str(hscale))

        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # Resize the current image displayed

        self.scale_factor = self.cal_scale(self.img)
        # print("New Scale: " + str(self.scale_factor))

        # img_sc = scipy.misc.imresize(self.img, self.scale_factor, interp='nearest')
        img_sc = np.uint8(self.img)
        self.width_offset = self.width - np.size(img_sc, axis=1)
        self.height_offset = self.height - np.size(img_sc, axis=0)
        # print("Offset, width: " + str(self.width_offset) + " height: " + str(self.height_offset) )
        img_can = Image.fromarray(img_sc)
        self.imgTk = ImageTk.PhotoImage(img_can)
        self.image_on_canvas = self.create_image(
            int(self.width / 2), int(self.height / 2), image=self.imgTk
        )
        self.itemconfig(self.image_on_canvas, image=self.imgTk)

        # XXX : How do we resize the ROI and row and column lines??
        # We might have to append all items on canvas, or just ignore for the moment
        return


# This is a blank matplotlib graph that we update with time
# Makes it easier to build guis
class Chrono_Plot(object):
    def __init__(self, master, n_pts, ylim, title):
        self.top = tk.Toplevel(master)
        self.top.wm_title(title)

        self.n_pts = n_pts
        self.ind = 0
        self.ch_plot = Figure(figsize=(5, 5), dpi=100)
        self.ch_axis = self.ch_plot.add_subplot(111)
        self.ch_plot_x = np.linspace(0, self.n_pts, self.n_pts)
        self.ch_plot_y = np.zeros((self.n_pts))
        (self.ch_line,) = self.ch_axis.plot(self.ch_plot_x, self.ch_plot_y, "-o")
        self.ch_axis.grid()
        self.ch_axis.set_ylim(ylim)
        self.ch_axis.set_title(title)
        self.plot_can = FigureCanvasTkAgg(self.ch_plot, self.top)
        # XXX : This is now depricated
        # self.plot_can.show()
        self.plot_can.draw()
        self.plot_can.get_tk_widget().grid(row=0, column=0, sticky=tk.S)
        return

    def update_plot(self, new_y, new_clim):
        # Update the data on the plot
        self.ch_plot_y = new_y
        self.ch_line.set_ydata(self.ch_plot_y)
        self.ch_axis.set_ylim(new_clim)
        # self.ch_line.set_ydata(self.ch_plot_y)
        self.plot_can.draw()
        return

    def update_point(self, new_pt, new_clim):
        # We update the current point
        self.ch_plot_y[self.ind] = new_pt
        self.ind = np.mod(self.ind + 1, self.n_pts)
        self.ch_line.set_ydata(self.ch_plot_y)
        self.ch_axis.set_ylim(new_clim)
        self.plot_can.draw()
        return


# This is just a blank canvas
# Want to add all the functionality to display time and space on subplots
# This is going to be a great way to view different frame types
class Chrono_Blank(object):
    def __init__(self, master, sz, title):
        # We generate a blank canvas to display an image on
        self.top = tk.Toplevel(master)
        self.top.wm_title(title)
        self.title = title
        # Allocate the
        arr = np.zeros(sz, dtype=np.uint8)
        tk.Grid.columnconfigure(self.top, 0, weight=1)
        tk.Grid.rowconfigure(self.top, 0, weight=1)
        # img = Image.fromarray(arr)
        # self.imgTk = ImageTk.PhotoImage(img)
        # self.can = tk.Canvas(self.top, width=sz[1],height=sz[0])
        self.can = Chrono_Canvas(
            self.top, width=sz[1], height=sz[0], bg="lightgrey", highlightthickness=0
        )
        self.top.bind("<Shift_L>", self.shift_press)
        self.top.bind("<KeyRelease-Shift_L>", self.shift_release)
        # Bind to "r" and "c" for plotting along the row and column
        self.top.bind("<c>", self.c_press)
        self.top.bind("<KeyRelease-c>", self.c_release)
        self.top.bind("<r>", self.r_press)
        self.top.bind("<KeyRelease-r>", self.r_release)
        self.can.bind("<ButtonPress-1>", self.mouse_press)
        self.can.bind("<ButtonRelease-1>", self.mouse_release)
        self.can.bind("<B1-Motion>", self.mouse_move)
        # self.image_on_canvas = self.can.create_image( sz[1]/2,sz[0]/2,image=self.imgTk)
        self.can.add_image(arr)
        self.frame_str = tk.StringVar()
        self.temp_str = tk.StringVar()

        self.frame_label = tk.Label(self.top, textvariable=self.frame_str)
        self.temp_label = tk.Label(self.top, textvariable=self.temp_str)
        # Labels for frame id and temperature
        self.can.grid(row=0, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.frame_label.grid(row=1, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.temp_label.grid(row=2, column=0, sticky=tk.S + tk.N + tk.W + tk.E)

        # Resize the relavent rows/columns

        self.but_clim = tk.Button(self.top, text="Color Limits", command=self.set_clim)
        self.but_clim.grid(row=3, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.clim_set = False
        self.clim = np.zeros(2, dtype=np.float)

        self.but_cm = tk.Button(self.top, text="Color Map", command=self.set_cm)
        self.but_cm.grid(row=4, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.cm = cm.gray

        self.shift_pressed = False
        self.mouse_pressed = False
        self.c_pressed = False
        self.col_line = False
        self.col_pt = 0
        self.r_pressed = False
        self.row_line = False
        self.row_pt = 0
        self.roi = [0, 0, 0, 0]
        self.rect_exists = False
        self.roi_update = False
        self.start_x = 0
        self.start_y = 0
        self.sz = sz
        return

    def set_clim(self):
        d = Chrono_Clim(self.top, self.clim)
        self.top.wait_window(d.top)
        new_clim = d.get_sel()
        if np.any(new_clim) == False:
            return
        self.clim[0] = new_clim[0]
        self.clim[1] = new_clim[1]
        self.clim_set = True

        return

    def set_cm(self):
        d = Chrono_Colormap(self.top)
        self.top.wait_window(d.top)
        self.cm = d.get_sel()
        return

    def mouse_press(self, event):
        # This mouse event is thanfully working
        # print('Clicked at ' + str(event.x) + ' ' + str(event.y) )
        self.mouse_pressed = True
        row = int(float(event.y - self.can.height_offset / 2) / self.can.scale_factor)
        col = int(float(event.x - self.can.width_offset / 2) / self.can.scale_factor)
        print("Row: " + str(row) + " Col: " + str(col))
        if self.shift_pressed == True:
            self.start_x = event.x
            self.start_y = event.y

        if self.c_pressed == True:
            if col < 0 or col >= self.sz[1]:
                return

            # We draw a line along the x axis
            if self.col_line == True:
                self.can.delete("col_line")
            else:
                # First time drawing
                plt_title = str(self.title) + " Col " + str(col)
                self.col_slice = Chrono_Plot(
                    self.top, self.sz[0], [0, 2 * np.pi], plt_title
                )
            self.can.create_line(
                event.x, 0, event.x, self.can.height, fill="red", tag="col_line"
            )
            self.col_pt = col
            self.col_line = True

        if self.r_pressed == True:
            # Make sure the row is on the canvas
            if row < 0 or row >= self.sz[0]:
                print("row " + str(row))
                print("sz[0] " + str(self.sz[0]))
                return

            if self.row_line == True:
                self.can.delete("row_line")
            else:
                print("create title")
                # XXX : This line is failing on python3
                plt_title = str(self.title) + " Row " + str(row)
                print("creating ChronoPlot")
                self.row_slice = Chrono_Plot(
                    self.top, self.sz[1], [0, 2 * np.pi], plt_title
                )
            # Make sure the row is on the canvas
            self.can.create_line(
                0, event.y, self.can.width, event.y, fill="green", tag="row_line"
            )
            self.row_pt = row
            self.row_line = True
        return

    def mouse_release(self, event):
        # print('Released at ' + str(event.x) + ' ' + str(event.y) )
        self.mouse_pressed = False
        # Delete the rectangle if it exists
        if self.rect_exists == True:
            # Grab the new ROI and use to update the ROI with time
            # print('Need to do ROI')
            if self.roi_update == False:
                plt_title = str(self.title) + " ROI"
                self.roi_plot = Chrono_Plot(self.top, 100, [0, 2 * np.pi], plt_title)
                self.roi_update = True
        return

    def c_press(self, event):
        # print("c pressed")
        self.c_pressed = True
        return

    def c_release(self, event):
        # print("c released")
        self.c_pressed = False
        return

    def r_press(self, event):
        self.r_pressed = True
        return

    def r_release(self, event):
        self.r_pressed = False
        return

    def mouse_move(self, event):
        if self.shift_pressed == True and self.mouse_pressed == True:
            if self.rect_exists == True:
                self.can.delete("rect")
            self.can.create_rectangle(
                self.start_x,
                self.start_y,
                event.x,
                event.y,
                width=1,
                outline="red",
                tag="rect",
            )
            start_row = int(
                float(self.start_y - self.can.height_offset / 2) / self.can.scale_factor
            )
            start_col = int(
                float(self.start_x - self.can.width_offset / 2) / self.can.scale_factor
            )
            end_row = int(
                float(event.y - self.can.height_offset / 2) / self.can.scale_factor
            )
            end_col = int(
                float(event.x - self.can.width_offset / 2) / self.can.scale_factor
            )

            # self.roi = [self.start_x,self.start_y,event.x,event.y]
            self.roi = [start_col, start_row, end_col, end_row]
            self.rect_exists = True

    def shift_press(self, event):
        # print('Shift Pressed')
        self.shift_pressed = True
        return

    def shift_release(self, event):
        # print('Shift Released')
        self.shift_pressed = False
        return

    def scale_img(self, img, clim):
        # We scale the entire image over the clim range
        # img_float = np.double(img)
        img_float = img
        img_float[np.isnan(img)] = 0
        img_float[img_float < clim[0]] = clim[0]
        img_float[img_float > clim[1]] = clim[1]

        img_float = (img_float - clim[0]) * (1 / (clim[1] - clim[0]))
        img_8bit = np.uint8(img_float * 255)
        return img_8bit

    def update_img(self, new_data, new_clim, frame_id):
        # Update the image on the canvas
        if self.clim_set == True:
            new_clim = self.clim
        else:
            self.clim = new_clim

        disp_img = self.scale_img(new_data, new_clim)
        cm_img = self.cm(disp_img)

        self.can.update_image(cm_img)
        self.frame_str.set("Frame: " + str(frame_id))
        # self.temp_str.set("Temperature: " + str(temperature))
        if self.col_line == True:
            if self.col_slice.top.winfo_exists() == 0:
                self.col_line = False
                self.c_pressed = False
                self.can.delete("col_line")
            slice_data = np.squeeze(new_data[:, self.col_pt])
            self.col_slice.update_plot(slice_data, new_clim)
        if self.row_line == True:
            if self.row_slice.top.winfo_exists() == 0:
                self.row_line = False
                self.r_pressed = False
                self.can.delete("row_line")
            slice_data = np.squeeze(new_data[self.row_pt, :])
            self.row_slice.update_plot(slice_data, new_clim)
        if self.roi_update == True:
            roi_data = np.reshape(
                np.squeeze(
                    new_data[self.roi[1] : self.roi[3], self.roi[0] : self.roi[2]]
                ),
                (-1),
            )
            roi_pt = np.nanmean(roi_data)
            self.roi_plot.update_point(roi_pt, new_clim)

        self.can.update()
        self.can.update_idletasks()
        # print("update_img")
        return


# Currently just returns the desired color map
class Chrono_Colormap(object):
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.var = tk.StringVar(self.top)

        # The dictionary that connects each item
        self.dic = {
            "viridis": cm.viridis,
            "plasma": cm.plasma,
            "inferno": cm.inferno,
            "magma": cm.magma,
            "jet": cm.jet,
            "gray": cm.gray,
            "hsv": cm.hsv,
            "seismic": cm.seismic,
            "gnuplot2": cm.gnuplot2,
            "bone": cm.bone,
            "copper": cm.copper,
            "hot": cm.hot,
            "spring": cm.spring,
            "autumn": cm.autumn,
            "winter": cm.winter,
        }

        # This if python 2 code
        if sys.version_info[0] == 2:
            kd = self.dic.keys()
            self.var.set(kd[0])
            self.option = apply(tk.OptionMenu, (self.top, self.var) + tuple(kd))
        else:
            kd = list(self.dic)
            self.var.set(kd[0])
            self.option = tk.OptionMenu(self.top, self.var, *kd)
        self.ok_but = tk.Button(self.top, text="Okay", command=self.ok)

        self.option.grid(row=0, column=0, sticky=tk.S + tk.N + tk.W + tk.E)
        self.ok_but.grid(row=1, column=0, sticky=tk.S + tk.N + tk.W + tk.E)

    def ok(self):
        self.top.destroy()
        # We want to return the function handle
        str_val = self.var.get()
        fun_ret = self.dic[str_val]
        return fun_ret

    def get_sel(self):
        str_val = self.var.get()
        return self.dic[str_val]


# Widget for updating the clim of plots
class Chrono_Clim(object):
    def __init__(self, master, curr_clim):
        top = self.top = tk.Toplevel(master)
        self.top.wm_title("Set CLIM")

        self.clim_entry = tk.Entry(top, text="CLIM")
        # Clean the entry
        self.clim_entry.delete(0, tk.END)
        self.clim_entry.insert(0, str(curr_clim[0]) + " " + str(curr_clim[1]))

        self.but_okay = tk.Button(top, text="Okay", command=self.ok)
        self.but_cancel = tk.Button(top, text="Cancel", command=self.cancel)

        self.but_okay.grid(row=2, column=0, sticky=tk.S)
        self.but_cancel.grid(row=2, column=1, sticky=tk.S)

        self.clim_entry.grid(row=0, column=0, columnspan=2)
        self.clim_entry.focus_set()

        return

    def ok(self):
        clim_str = self.clim_entry.get()
        clim_list = clim_str.split()
        if len(clim_list) != 2:
            self.sel_item = np.zeros(2, dtype=np.float)
            self.top.destroy()
            return

        self.sel_item = np.zeros(2, dtype=np.float)
        for n in range(0, len(clim_list)):
            self.sel_item[n] = float(clim_list[n])
        # self.sel_item = float(clim_list)
        self.top.destroy()
        return

    def cancel(self):
        self.sel_item = []
        self.top.destroy()
        return

    def get_sel(self):
        return self.sel_item
