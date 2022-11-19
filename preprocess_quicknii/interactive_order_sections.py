import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import skimage
import pandas as pd
from tkinter import Tk, Canvas, Frame, Label, Entry, Button, filedialog, StringVar#, PhotoImage

import image_processing_utils as u2

# interactive view imports
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from pylab import get_current_fig_manager

def clear_axis(an_ax):
    an_ax.set_frame_on(False)
    an_ax.get_yaxis().set_ticks([])
    an_ax.get_yaxis().set_ticklabels([])
    an_ax.get_xaxis().set_ticks([])
    an_ax.get_xaxis().set_ticklabels([])
    an_ax.set_xlabel('')
    an_ax.set_ylabel('')

def get_primary_monitor_screensize():
    '''returns a tuple of width, height for the primary monitor'''
    import ctypes
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return (screensize)

def get_tiled_frame_geometry(i, n_cols, x, wfs = 275, h_spacing=10, v_spacing=50, v_offset=40):
    '''return a tup of (x-pos, y-pos, width, height)'''
    # adjust defaults if monitor is smaller
    if get_primary_monitor_screensize() == (1920, 1080):
        wfs, h_spacing, v_spacing, v_offset = 250, 2, 5, 30

    geometry=((wfs+h_spacing)*(i%n_cols), x*(wfs+v_spacing)+v_offset,wfs,wfs)
    return geometry
    
def get_row_y_thresholds():
    '''scale the criteria for determining which row an image is in for smaller screens'''
    if get_primary_monitor_screensize() == (1920, 1080):
        return [0, 250, 500, 750]
    else:
        return [0, 300, 650, 950]
            

# set window extent and name
def config_window (title=None, geometry=None):
    # default display geometry
    if not geometry:
        geometry= (0, 50,1700,900)

    # set window extent
    thismanager = get_current_fig_manager()
    thismanager.window.setGeometry(*geometry)
    thismanager.set_window_title(title)
    
    # return the manager to keep track
    return thismanager

class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print (event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def make_interactive_ax(an_ax):
    zp = ZoomPan()
    figZoom = zp.zoom_factory(an_ax, base_scale = 1.1)
    figPan = zp.pan_factory(an_ax)
    return zp

def print_ax_list(ax_list):
    for ax_dict in ax_list:
        manager = ax_dict['manager']
        print(Path(ax_dict['image_path']).stem, '--->', manager.window.x() , manager.window.y())


def get_position_dicts(ax_list, n_rows, base_dir,img_dirs):
    # once all images have been rearranged run this
    pos_dicts = [{'x':ax_dict['manager'].window.x() , 'y':ax_dict['manager'].window.y(), 'p':ax_dict['image_path']} for ax_dict in ax_list]
    pos_df = pd.DataFrame(pos_dicts)
    print(pos_df.head())

    # sort by coordinates
    ordered_fps = []
    row_dict_groups = dict(zip(range(n_rows), get_row_y_thresholds()))
    row_dict = dict(zip(range(n_rows), [{},{},{},{}]))
    # break up into rows
    for row_i, row in pos_df.iterrows():
        x,y = row['x'], row['y']
        for rn, thresh in row_dict_groups.items():
            if int(y) in range(thresh-130, thresh + 130):
                row_dict[rn][x] = {'y':y, 'p':row['p']}
    
    for row_i, img_row in row_dict.items():
        sorted_row = sorted(img_row)
        for el in sorted_row:
            img_path = Path(img_row[el]['p']).stem
            ordered_fps.append(img_path)
    
    print(ordered_fps)

    out_df = pd.DataFrame(columns=['ImageFolder', 'Order', 'rotate', 'h_flip', 'remove'])
    out_df['Order'] = ordered_fps
    out_df['ImageFolder'] = Path(base_dir).parent.name
    
    assert len(ordered_fps) == len(img_dirs)
    print('checking num images match expected and ordered\n', f'expected: {len(img_dirs)}', f'num_windows: {len(ordered_fps)}')
    return out_df




'''
#################################################################################################
MAIN
#################################################################################################
'''

# base_dir = r'D:\TEL Slides\ScriptOutput2\resized_images\TEL5\Merged'
animal_index_i = 14
base_dir = os.path.join(r'D:\TEL Slides\ScriptOutput2\resized_images', f'TEL{animal_index_i}', 'Merged')
base_dir_filter = '.png'
img_dirs = sorted([os.path.join(base_dir, el) for el in os.listdir(base_dir) if base_dir_filter in el])

# try to organize in 4 rows 
n_rows = 4
n_cols = math.ceil(len(img_dirs)/n_rows)

# fig, axs = plt.subplots(n_rows, n_cols, figsize=(40,20))
ax_list = []
i = 0

for x in range(n_rows):
    for y in range(n_cols):
        if i == len(img_dirs):
            break
        fig, ax = plt.subplots(figsize=(10,10))
        img_path = img_dirs[i]
        img = skimage.io.imread(img_path)
        ax.imshow(img)

        # format axis
        clear_axis(ax)
        ax.set_title('_'.join(el for el in Path(img_path).stem.split('_')[:2]))
        ax.margins(x=0., y=0.)
        
        # get interactive figure manager
        current_manager = config_window(f'{Path(img_path).stem}', geometry=get_tiled_frame_geometry(i, n_cols, x))
        # store all managers
        ax_list.append({'manager':current_manager, 'image_path':img_path})

        # increment image
        i+=1
        # interactive stuff
        zp = make_interactive_ax(ax)

print_ax_list(ax_list)

out_df = get_position_dicts(ax_list, n_rows, base_dir,img_dirs)

# run this after everything looks good
if bool(0):
    out_df = get_position_dicts(ax_list, n_rows, base_dir,img_dirs)
    df_out_path = os.path.join(Path(base_dir).parent, 'ImageOrder.xlsx')
    out_df.to_excel(df_out_path, index=False)

if bool(0):
    plt.close('all')



    



        







