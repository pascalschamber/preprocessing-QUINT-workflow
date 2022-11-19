import os
from skimage import io
import numpy as np
import javabridge as jb
import bioformats as bf
import matplotlib.pyplot as plt
from pathlib import Path

def separate_channels(img):
    ''' separate channels '''
    return [img[..., ch_key] for ch_key in range(img.shape[2])]

def normalize(img):
    ''' for each channel, normalize px intensity values into range [0,1] '''
    new_img = np.zeros_like(img.astype('float16'))
    for ch_key in range(img.shape[2]):
        ch_slc = img[..., ch_key]
        ch_slc = ch_slc/ch_slc.max()
        new_img[..., ch_key] = ch_slc
    return new_img

def threshold(img, thresh_dict):
    ''' min/max thresholding - dict provides min/max for each channel '''
    if thresh_dict == None:
        return img

    new_img = np.zeros_like(img)
    for ch_key, thresh_minMax in thresh_dict.items():
        # get ch slice
        ch_slc = img[..., ch_key]
        # thresh max
        ch_slc = np.where(ch_slc < thresh_minMax[1], ch_slc, thresh_minMax[1])
        # thresh min
        ch_slc = np.where(ch_slc > thresh_minMax[0], ch_slc, 0)
        # set ch_slc on new_img
        new_img[..., ch_key] = ch_slc
    return new_img

def BGR_to_RGB(img):
    '''BGR to RGB'''
    new_img = img.copy()
    new_img[..., 0] = img[..., 2]
    new_img[..., 2] = img[..., 0]
    return new_img

def img_stats(img):
    '''output img info'''
    print(f'shape: {img.shape}\nmax: {img.max()} min: {img.min()}\ndtype: {img.dtype}')

def get_oXML_image_metadata(oXML_obj):
    '''iterate through series, getting metadata for each'''
    image_dims = {}
    for i in range(oXML_obj.get_image_count()):
        image_dims[i] = {}
        image_dims[i]['X'] = oXML_obj.image(i).Pixels.SizeX 
        image_dims[i]['Y'] = oXML_obj.image(i).Pixels.SizeY
        image_dims[i]['date'] = oXML_obj.image(i).get_AcquisitionDate()
        image_dims[i]['px_type']= oXML_obj.image(i).Pixels.get_PixelType()
    return image_dims

def plot_imgs (img_list, SEPARATE=False, PLOT=True):
    '''
        # images are 16-bit, max px value of 64,000
        # slice a 3-ch image with [:, :, 0]
        # image channel order is BGR
    '''
    if not PLOT:
        return

    for im in img_list:
        
        # force to float img
        if im.max() != 1.0:
            im = im.astype('float32') # dtype float16 is unsupported with imshow
            im = im/im.max()
        
        if SEPARATE:
            fig, axs = plt.subplots(3,1, figsize=(20,10))
            axs[0].imshow(im[:, :, 0], vmin=0., vmax=1.0, cmap='Reds')
            axs[1].imshow(im[:, :, 1], vmin=0., vmax=1.0, cmap='Greens')
            axs[2].imshow(im[:, :, 2], vmin=0., vmax=1.0, cmap='Blues')
        else:
            fig, axs = plt.subplots(1,1, figsize=(20,10))
            axs.imshow(im, vmin=0, vmax=1.0)
        plt.show()

def show(image, def_title=''):
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=1.0)
    fig.suptitle(def_title)
    plt.show()

def image_histogram(image):
    ''' plot the histogram of an image'''
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(image, bins=255)
    ax.set_xlim(0, image.max())
    plt.show()

def save_images(image_path, out_dir, imgs, suffix_suffix='_S'):
    ''' save images - imgs must be an iterable '''
    assert hasattr(imgs, '__iter__'), f'{type(imgs)} is not iterable, pass imgs as a list'
    for i in range(len(imgs)):
        img = imgs[i]
        suffix = f'ch-{i}'
        if len(imgs) == 1: suffix = ''

        out_name = f'{Path(image_path).stem}_{suffix}{suffix_suffix}.tif'
        out_path = os.path.join(out_dir, out_name)
        io.imsave(out_path, img)

def get_files_from_nested_dir(base_dir, base_dir_filter='', file_type=True):
    '''iterate through folders in a folder pulling out file paths'''
    supdirs = sorted([os.path.join(base_dir, el) for el in os.listdir(base_dir) if base_dir_filter in el])
    all_paths = []
    for cdir in supdirs:
        all_paths.extend(sorted([os.path.join(cdir, el) for el in os.listdir(cdir) if el[-4:]==file_type]))
    return all_paths

def convert_vsi(
    base_dir, 
    out_dir_base, 
    series_index, 
    FILE_TYPE='.vsi',
    DIR_FILTER_STR='',  
    image_fn_skip_string='',
    thresh_dict=None,
    SEPARATE_CHANNELS=True,
    SAVE_SEPARATE_CHANNELS=True,
    SAVE_IMAGE=True,
    PLOT=False,
    PLOT_SEPARATE=False,
    ):
    '''
    ###########################################################################################
    Description
    ~~~~~~~~~~~
        convert Olympus .vsi image files to .png, or any other format supported by skimage
        extract and save individual channels 
        threshold individual channels

        .vsi files are not supported by skimage, so need to use java virtual machine to run imageJ
    
    PARAMS
    ~~~~~~~~~~~
        base_dir: directory containing .vsi files
        out_dir_base: output directory
        DIR_FILTER_STR = only look at folders that contain this string in thier name
        image_fn_skip_string: skip images that have this str in thier filename
        series_index: int: def which image to get (usually contains many different resolutions, 0 would be highest)
        thresh_dict: dict: define thresholding for each channel 
            e.g. 
            {   0:[0, 20000],
                1:[0, 20000],
                2:[0, 20000],}
        SEPARATE_CHANNELS: bool: split channels
        SAVE_SEPARATE_CHANNELS: bool: save split channels
        SAVE_IMAGE: bool: save merged image
        PLOT: bool: view the output
        PLOT_SEPARATE: bool: view the split channels output, else view merged

    ###########################################################################################
    '''
    all_paths = get_files_from_nested_dir(base_dir, base_dir_filter=DIR_FILTER_STR, file_type=FILE_TYPE) 

    # make base output dir
    if not os.path.exists(out_dir_base): os.mkdir(out_dir_base)

    jb.start_vm(class_path=bf.JARS)
    for path_index in list(range(len(all_paths))):
        
        image_path = all_paths[path_index]
        
        out_dir = os.path.join(out_dir_base, Path(image_path).parent.stem)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        # skip these image files
        if image_fn_skip_string in Path(image_path).stem:
            continue

        myXML = bf.get_omexml_metadata(image_path)
        oXML = bf.OMEXML(myXML)
        num_images = oXML.get_image_count()
        image_dims = get_oXML_image_metadata(oXML)
        print(f'#{path_index}: {image_path}')
        print(f'number of series: {num_images}')
        print(f'image dims: {image_dims}')


        with bf.ImageReader(image_path) as reader:
            img = reader.read(series=series_index,
                rescale=False,
            )

        img_stats(img)
        assert image_dims[0]['px_type'] == 'uint16'
        assert img.shape[2] == 3

        new_img = threshold(img, thresh_dict)
        new_img = BGR_to_RGB(new_img)
        new_img = normalize(new_img)

        if SEPARATE_CHANNELS:
            ch_imgs = separate_channels(new_img)
            if SAVE_SEPARATE_CHANNELS:
                save_images(image_path,out_dir, ch_imgs)

        if SAVE_IMAGE:
            save_images(image_path,out_dir, [new_img])
            
        # plot if enabled
        plot_imgs([img, new_img], PLOT=PLOT, SEPARATE=PLOT_SEPARATE)

    jb.kill_vm()




'''
##############################################################################################
User input
##############################################################################################
'''

if __name__ == '__main__':

    base_dir = r'F:\TEL Slides\Batch #7'
    out_dir_base = r'F:\TEL Slides\ScriptOutput2'
    DIR_FILTER_STR = 'Slide' # only look at folders that contain this string in thier name
    image_fn_skip_string = 'Overview'

    # def which image to get (usually contains many different resolutions, 0 would be highest)
    series_index = 0
    # define thresholding for each channel
    thresh_dict = {
        0:[0, 20000],
        1:[0, 20000],
        2:[0, 20000],
    }

    SEPARATE_CHANNELS = bool(1)
    SAVE_SEPARATE_CHANNELS = bool(1)
    SAVE_IMAGE = bool(1)
    PLOT = bool(0)
    PLOT_SEPARATE = bool(0)
    
    convert_vsi(
        base_dir, 
        out_dir_base, 
        series_index, 
        FILE_TYPE='.vsi',
        DIR_FILTER_STR=DIR_FILTER_STR,  
        image_fn_skip_string=image_fn_skip_string,
        thresh_dict=thresh_dict,
        SEPARATE_CHANNELS=SEPARATE_CHANNELS,
        SAVE_SEPARATE_CHANNELS=SAVE_SEPARATE_CHANNELS,
        SAVE_IMAGE=SAVE_IMAGE,
        PLOT=PLOT,
        PLOT_SEPARATE=PLOT_SEPARATE,
    )






