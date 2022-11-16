def read_image(path_to_image):
    ''' read an image using simpleitk plugin to avoid decompression bomb warnings when opening large files '''
    import skimage
    return skimage.io.imread(path_to_image, plugin='simpleitk')

def show(image, def_title='none', size=(15, 15), cmap=None, SAVE_PATH=False):
    import matplotlib.pyplot as plt
    
    if isinstance(image, tuple): 
        raise ValueError (f'show cannot handle inputs of type tuple, ensure array is passed')
    
    # read a 3d array or colored 3d array
    if image.ndim == 3 or (image.ndim ==4 and image.shape[-1] == 3):
        if image.ndim == 3 and image.shape[-1] == 3: # for 3 dim image where last dim is RGB
            pass
        image = create_MIP(image)[0]
    elif image.ndim == 2:
        pass
    else:
        raise ValueError (f'cannot interpret image of shape {image.shape}')
    
    # plot image
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(image, cmap=cmap)
    if def_title != 'none':
        ax.set(title=(str(def_title)))
    
    # save image
    if SAVE_PATH:
        fig.savefig(SAVE_PATH, bbox_inches='tight', dpi=300)

    plt.show()

def create_MIP(array, axis='z'):
    import numpy as np
    assert array.ndim > 2, f"array is not 3D, it has {array.ndim} dimensions"
    meta_slice_px_average = []
    meta_slice_px_min = []
    meta_slice_px_max = []
    meta_slice_px_sum = []
    try:
        if axis == 'z':
            max_ = np.zeros_like(array[0, ...])
            depth = array.shape[0]
            for i in range(0, depth):
                image_slice = array[i, ...]
                max_ = np.maximum(max_, image_slice)
                meta_slice_px_average.append(np.average(image_slice))
                meta_slice_px_min.append(image_slice.min())
                meta_slice_px_max.append(image_slice.max())
                meta_slice_px_sum.append(image_slice.sum())

        if axis == 'y':
            max_ = np.zeros_like(array[:, 0, :])
            max_ = np.rot90(max_)
            depth = array.shape[1]
            for i in range(0, depth):
                image_slice = array[:, i, :]
                image_slice = np.rot90(image_slice)
                max_ = np.maximum(max_, image_slice)
                meta_slice_px_average.append(np.average(image_slice))
                meta_slice_px_min.append(image_slice.min())
                meta_slice_px_max.append(image_slice.max())
                meta_slice_px_sum.append(image_slice.sum())

        if axis == 'x':
            max_ = np.zeros_like(array[:, :, 0])
            max_ = np.rot90(max_)
            depth = array.shape[2]
            for i in range(0, depth):
                image_slice = array[:, :, i]
                image_slice = np.rot90(image_slice)
                max_ = np.maximum(max_, image_slice)
                meta_slice_px_average.append(np.average(image_slice))
                meta_slice_px_min.append(image_slice.min())
                meta_slice_px_max.append(image_slice.max())
                meta_slice_px_sum.append(image_slice.sum())

        meta_slice_px_vals = {'average_px_value':meta_slice_px_average,
         'min_px_value':meta_slice_px_min,
         'max_px_value':meta_slice_px_max,
         'sum_px_values':meta_slice_px_sum}
        meta_whole_image_px_vals = {'average_px_value':np.average(meta_slice_px_average),
         'min_px_value':min(meta_slice_px_min),
         'max_px_value':max(meta_slice_px_max),
         'sum_px_values':sum(meta_slice_px_sum)}
        return (
         max_, meta_slice_px_vals, meta_whole_image_px_vals)
    except:
        print('shape = ', array.shape)
        print('min/max = ', array.min(), array.max())
        print('dtype = ', array.dtype)
        raise IndexError