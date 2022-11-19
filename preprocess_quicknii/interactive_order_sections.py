import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import skimage
import pandas as pd
from tkinter import Tk, Canvas, Frame, Label, Entry, Button, filedialog, StringVar#, PhotoImage
import ctypes

import image_processing_utils as u2

# interactive view imports
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from pylab import get_current_fig_manager


'''
#################################################################################################
    Interactive image display
#################################################################################################
'''
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
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    #!!! overwriting for testing dimension mismatch, gets resolution correctly but in display appears rending is smaller
    return (2050,1100)
    return (screensize)

# def get_tiled_frame_geometry(row_i, n_cols, row_x, wfs_x = 275, wfs_y=275, h_spacing=10, v_spacing=50, v_offset=40):
def get_tiled_frame_geometry(row_i, n_cols, row_x, wfs_x, wfs_y, h_spacing, v_spacing, v_offset):
    '''return a tup of (x-pos, y-pos, width, height)'''
    geometry=(
        (wfs_x+h_spacing)*(row_i%n_cols), 
        row_x*(wfs_y+v_spacing)+v_offset,
        wfs_x,
        wfs_y
    )
    return geometry
    
def get_row_y_thresholds(final_img_px_size_y, n_rows):
    '''scale the criteria for determining which row an image is in for smaller screens'''
    return [(final_img_px_size_y+h_spacing+v_offset)*iii for iii in range(n_rows)]


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



'''
#################################################################################################
    State variables class 
#################################################################################################
'''
class StateVars:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ax_list, self.n_rows, self.base_dir, self.img_dirs = None, None, None, None

    def update_state(self):
        self.ax_list, self.n_rows, self.base_dir, self.img_dirs = ax_list, n_rows, base_dir, img_dirs

    def print_ax_list(self):
        self.update_state()
        for i, ax_dict in enumerate(self.ax_list):
            if i%n_cols == 0:                print()
            manager = ax_dict['manager']
            print(Path(ax_dict['image_path']).stem, '--->', manager.window.x() , manager.window.y())

    def print_state(self):
        self.update_state()
        self.print_ax_list()

    def get_position_dicts(self):
        self.update_state()
        # once all images have been rearranged run this
        pos_dicts = [{'x':ax_dict['manager'].window.x() , 'y':ax_dict['manager'].window.y(), 'p':ax_dict['image_path']} for ax_dict in self.ax_list]
        pos_df = pd.DataFrame(pos_dicts)
        print(pos_df.head())

        # sort by coordinates
        ordered_fps = []
        row_dict_groups = dict(zip(range(self.n_rows), get_row_y_thresholds(final_img_px_size_y, self.n_rows)))
        row_dict = dict(zip(range(self.n_rows), [{},{},{},{}]))
        # break up into rows
        for row_i, row in pos_df.iterrows():
            xx,yy = row['x'], row['y']
            for rn, thresh in row_dict_groups.items():
                if int(yy) in range(thresh-130, thresh + 130):
                    row_dict[rn][xx] = {'y':yy, 'p':row['p']}
        self.row_dict = row_dict

        for row_i, img_row in row_dict.items():
            sorted_row = sorted(img_row)
            for el in sorted_row:
                img_path = Path(img_row[el]['p']).stem
                ordered_fps.append(img_path)
        self.ordered_fps = ordered_fps
        print(ordered_fps)

        out_df = pd.DataFrame(columns=['ImageFolder', 'Order', 'rotate', 'h_flip', 'remove'])
        out_df['Order'] = ordered_fps
        out_df['ImageFolder'] = Path(self.base_dir).parent.name
        
        assert len(ordered_fps) == len(self.img_dirs)
        print('checking num images match expected and ordered\n', f'expected: {len(self.img_dirs)}', f'num_windows: {len(ordered_fps)}')
        return out_df
    
    def save_state_to_xlsx(self):
        out_df = self.get_position_dicts()
        df_out_path = os.path.join(Path(self.base_dir).parent, 'ImageOrder.xlsx')
        out_df.to_excel(df_out_path, index=False)

    

'''
#################################################################################################
    UI for on screen buttons 
#################################################################################################
'''  
def init_UI():
    HEIGHT = 150
    WIDTH = 500
    root = Tk()
    root.title('Order Sections')
    canvas = Canvas(root, height=HEIGHT, width=WIDTH)
    canvas.pack() 
    frame = Frame(root, bg='#34ebde', bd=5)
    frame.place(relx=0.025, rely=0.05, relwidth=0.95, relheight=0.90)
    frame_label = Label(frame,
                        text='Controls',
                        anchor = 'center', bg='white'
                        )
    frame_label.place(relx = 0, rely = 0, relwidth=1, relheight=0.2)
    return root, canvas, frame

def build_UI(root, frame, filename, state_vars_object):
    make_directory_button(root, frame, filename)
    make_check_screen_button(frame, state_vars_object)
    make_save_order_button(frame, state_vars_object)
    make_close_window_button(frame)
    make_get_next_directory_button(frame)

# buttons
###############################################
# get directory
def make_directory_button(root, frame, filename):
    label_entry1 = Label(frame, 
                     text='images_directory: ', 
                     bg='#34ebde', anchor = 'center')
    label_entry1.place(relx=0, rely=0.25, relwidth=0.25, relheight= 0.2) 

    def prompt_directory():
        root.filename = str(filedialog.askopenfilename())
        dir_label = Label(frame, text=root.filename, bg='#34ebde', anchor = 'w')
        dir_label.place(relx=0.25, rely=0.25, relwidth=0.55, relheight= 0.2)

    root.filename=base_dir # current file name
    dir_label = Label(frame, text=root.filename, bg='#34ebde', anchor = 'w')
    dir_label.place(relx=0.25, rely=0.25, relwidth=0.55, relheight= 0.2)

    directory_button = Button(
                            frame, text='Browse', bg='white',
                            command=lambda : prompt_directory()
                            )
    directory_button.place(relx=0.85, rely=0.25, relwidth=0.15, relheight= 0.2)

# button to check order matches screen
########################################
def make_check_screen_button(frame, state_vars_object):       
    check_screen_button = Button(
        frame, text='check order', bg='white', 
        command=lambda : state_vars_object.print_state())
    check_screen_button.place(relx=0.00, rely=0.5, relwidth=0.2, relheight= 0.2)

# button to save order to excel
########################################
def make_save_order_button(frame, state_vars_object):
    save_order_button = Button(
        frame, text='save order', bg='white', 
        command=lambda : state_vars_object.save_state_to_xlsx())
    save_order_button.place(relx=0.25, rely=0.5, relwidth=0.2, relheight= 0.2)

# button to close all images
########################################
def make_close_window_button(frame):
    close_windows_button = Button(
        frame, text='close all', bg='white', 
        command=lambda : plt.close('all'))
    close_windows_button.place(relx=0.0, rely=0.75, relwidth=0.15, relheight= 0.2)

# button to get next directory
########################################
def make_get_next_directory_button(frame):
    def get_next_directory():
        plt.close('all')
        pass
    get_next_directory_button = Button(
        frame, text='get next directory', bg='white', 
        command=lambda : get_next_directory())
    get_next_directory_button.place(relx=0.2, rely=0.75, relwidth=0.30, relheight= 0.2)

'''
#################################################################################################
MAIN
#################################################################################################
'''

# base_dir = r'D:\TEL Slides\ScriptOutput2\resized_images\TEL5\Merged'
animal_index_i = 1
base_dir = os.path.join(r'C:\Users\pasca\Desktop', f'TEL{animal_index_i}', 'Merged')
base_dir_filter = '.png'
img_dirs = sorted([os.path.join(base_dir, el) for el in os.listdir(base_dir) if base_dir_filter in el])

# init state
state_obj = StateVars(base_dir)

# try to organize in 4 rows 
n_rows = 4
n_cols = math.ceil(len(img_dirs)/n_rows)


# create UI
root, canvas, frame = init_UI()
# add buttons
build_UI(root, frame, base_dir, state_obj)



ax_list = []
img_dirs_i = 0
for x_row in range(n_rows):
    for y_row in range(n_cols):
        if img_dirs_i == len(img_dirs):
            break
        
        # plot_images()
        # def plot_images():
        fig, ax = plt.subplots(figsize=(10,10))
        img_path = img_dirs[img_dirs_i]
        img = skimage.io.imread(img_path)
        ax.imshow(img)

        # format axis
        clear_axis(ax)
        ax.set_title('_'.join(el for el in Path(img_path).stem.split('_')[:2]))
        ax.margins(x=0., y=0.)
        
        # set img size based on screen resolution
        def get_img_px_sizes (n_rows, n_cols, h_spacing, v_spacing, v_offset):
            n_rows = 4
            n_cols = math.ceil(len(img_dirs)/n_rows)
            mx, my = get_primary_monitor_screensize()
            img_px_x = int(mx/n_cols)
            img_px_y = int(my/n_rows)
            final_img_px_size_x = img_px_x - h_spacing #- ((n_cols-1) * h_spacing)
            final_img_px_size_y = img_px_y - v_spacing
            return (final_img_px_size_x, final_img_px_size_y)

        h_spacing, v_spacing, v_offset = 10, 50, 40
        final_img_px_size_x, final_img_px_size_y = get_img_px_sizes (n_rows, n_cols, h_spacing, v_spacing, v_offset)


        # get interactive figure manager
        current_manager = config_window(
            title = f'{Path(img_path).stem}', 
            geometry=get_tiled_frame_geometry(
                img_dirs_i, 
                n_cols, 
                x_row, 
                final_img_px_size_x, 
                final_img_px_size_y, 
                h_spacing, v_spacing, v_offset)
            )
        # store all managers
        ax_list.append({'manager':current_manager, 'image_path':img_path})

        # increment image
        img_dirs_i+=1
        # interactive stuff
        zp = make_interactive_ax(ax)

state_obj.print_ax_list()
out_df = state_obj.get_position_dicts() # alternative for viewing details


# if not using UI, run this after everything looks good
if bool(0):
    state_obj.save_state_to_xlsx()

if bool(0):
    plt.close('all')

''' end main loop '''
root.mainloop()

#TODO for some reason it thinks my monitor is like 2000, by 1000


    



        







