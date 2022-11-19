import os
import skimage
from PIL import Image, ImageEnhance
import re
from pathlib import Path
from collections import Counter
import image_processing_utils as u2
import numpy as np

def get_files_from_2nested_dir(base_dir, base_dir_filter='', file_type=True):
    '''iterate through folders in a folder pulling out file paths'''
    supdirs = sorted([os.path.join(base_dir, el) for el in os.listdir(base_dir) if base_dir_filter in el])
    all_paths = []
    for cdir in supdirs:
        all_paths.extend(sorted([os.path.join(cdir, el) for el in os.listdir(cdir)]))
    return all_paths

def ensure_unique_names(dict_ch_image_paths, animal):
    for k,v in dict_ch_image_paths.items():
        v_stem = [el.stem for el in v]
        if not len(v_stem) == len(set(v_stem)):
            
            print( f'duplicate found in {k}')
            count = Counter(v_stem)
            errors_count =[]
            for el, c in count.items():
                if c >1 :
                    errors_count.append([el,c])
            raise ValueError(animal, errors_count)

# renaming to give unique name to each
def renaming_image_0x():
    if bool(0):
        to_rename = [
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_01_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_02_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_04_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_05_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_06_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_07_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_08_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_09_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_10_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_11_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_12_ch-1_S.tif'),
            Path('D:/TEL Slides/ScriptOutput/TEL12 - 2 out of 4/Histone-EGFP/Image_01_ch-1_S.tif'),
        ]
        ch_indexes = [1, 2, 0]
        for i, ch in enumerate(['Histone-EGFP', 'Hoechst', 'Zif-268']):
            replace_ch = str(ch_indexes[i])
            replace_with = '3'
            for el in to_rename:
                old_path = os.path.join(el.parent.parent, ch, el.name)
                s = el.stem 
                st_ind = s.find('Image_0') + len('Image_0')
                new_s = s[:st_ind] + replace_with + s[st_ind+1:] + el.suffix
                new_ch_ind = new_s.find('ch-')+ len('ch-')
                new_s = new_s[:new_ch_ind] + replace_ch + new_s[new_ch_ind+1:]
                new_path = os.path.join(el.parent.parent, ch, new_s)

                old_path = os.path.join(el.parent.parent, ch, s[:new_ch_ind] + replace_ch + s[new_ch_ind+1:] + el.suffix)
                if not os.path.exists(old_path): raise ValueError(old_path)
                os.rename(old_path, new_path)


# image conversion from tif to jpg
def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8)


root_dir = r'D:\TEL Slides\ScriptOutput'
output_dir = r'D:\TEL Slides\ScriptOutput2'
base_dir_filter=''

animal_base_name = 'TEL'
num_animals = 14
skip_dirs = ['TEL1', 'TEL2', 'TEL3', 'TEL4']

channel_names_list = ['Histone-EGFP', 'Hoechst', 'Zif-268']
channel_order_list = [1,2,0]

RESIZE = 0.10

def get_directories(root_dir, output_dir, base_dir_filter=''):
    '''get all animal dirs and init output folder'''
    base_dir = root_dir
    # init
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    # make dirs to store images that will be created
    large_image_dir = os.path.join(output_dir, 'large_images')
    if not os.path.exists(large_image_dir): os.mkdir(large_image_dir)
    resize_image_dir = os.path.join(output_dir, 'resized_images')
    if not os.path.exists(resize_image_dir): os.mkdir(resize_image_dir)
    # get all folders
    supdirs = sorted([Path(os.path.join(base_dir, el)) for el in os.listdir(base_dir) if base_dir_filter in el])
    return supdirs, large_image_dir, resize_image_dir

def get_animals(supdirs, animal_base_name, num_animals):
    '''group directories by animal'''
    animals = {}
    for i in range(1,num_animals+1):
        pattern = f'{animal_base_name}{i} '
        animals[pattern[:-1]] = []
        for dir in supdirs:
            match = re.search(pattern, dir.stem)
            if match:
                animals[pattern[:-1]].append(dir)
    return animals

def remove_animals(animals, remove_list):
    '''skip already completed ones'''
    if remove_list != []:
        for comp in remove_list:
            del animals[comp]
    return animals

# TODO configure arguments
def merge_and_resize(
    root_dir, output_dir, channel_names_list, channel_order_list, 
    RESIZE=RESIZE,
    base_dir_filter='',

    ):
    

    supdirs, large_image_dir, resize_image_dir = get_directories(root_dir, output_dir, base_dir_filter=base_dir_filter)
    animals = remove_animals(get_animals(supdirs, animal_base_name, num_animals), skip_dirs)
    
    # combine slides
    for animal, dirs in animals.items():
        print(f'processing animal {animal}')
        # get all images for each channel by animal
        dict_ch_image_paths = {k:[] for k in channel_names_list}
        for slide_dir in dirs:
            ch_dirs = sorted([Path(os.path.join(slide_dir, el)) for el in os.listdir(slide_dir)])
            for ch_dir in ch_dirs:
                ch_image_paths = sorted([Path(os.path.join(ch_dir, el)) for el in os.listdir(ch_dir)])
                dict_ch_image_paths[ch_dir.stem].extend(ch_image_paths)

        # ensure unqiue names
        ensure_unique_names(dict_ch_image_paths, animal)
        
        # for each animal make dirs for large images and resized images
        for adir in [large_image_dir, resize_image_dir]:
            if not os.path.exists(os.path.join(adir, animal)): os.mkdir(os.path.join(adir, animal))

        # init merge
        resized_image_merge_dict = {}
        ch_indicies = dict(zip(channel_names_list, channel_order_list))
        # combine all folders now that all names are unique
            # and resize them and init merge
        
        for ch_name, og_paths in dict_ch_image_paths.items():
            large_ch_path_dir = os.path.join(large_image_dir, animal, ch_name)
            if not os.path.exists(large_ch_path_dir): os.mkdir(large_ch_path_dir)
            resized_ch_path_dir = os.path.join(resize_image_dir, animal, ch_name)
            if not os.path.exists(resized_ch_path_dir): os.mkdir(resized_ch_path_dir)

            for og_path in og_paths:
                # outname, add the s000 suffix
                out_name = f'{og_path.stem[:-1]}s000.png'
                # remove the _ joining image id
                match = re.search('Image_\d\d_\d\d_', out_name)
                if match:
                    out_name = out_name[:len('Image_xx')] + out_name[len('Image_xx')+1:]

                # outnames
                large_out_path = os.path.join(large_ch_path_dir, out_name)
                resized_out_path = os.path.join(resized_ch_path_dir, out_name)
                
                # create a dict for each image that points to individual channels
                merge_match = re.search('Image_\d+_', out_name)
                if not merge_match:
                    raise ValueError(out_name)
                base_name = out_name[:merge_match.span()[1]] + 'merged_s000.png'
                
                if base_name not in resized_image_merge_dict:
                    resized_image_merge_dict[base_name] = {0:'', 1:'', 2:''}

                resized_image_merge_dict[base_name][ch_indicies[ch_name]] = {
                    'old_path':og_path,
                    'resized_out_path':resized_out_path,
                    'large_out_path':large_out_path,
                }


        # merge and read at same time
        ############################################################################
        resized_merge_out_dir = os.path.join(resize_image_dir, animal, 'Merged')
        if not os.path.exists(resized_merge_out_dir): os.mkdir(resized_merge_out_dir)
        large_merge_out_dir = os.path.join(large_image_dir, animal, 'Merged')
        if not os.path.exists(large_merge_out_dir): os.mkdir(large_merge_out_dir)

        print(f'\tresizeing images for animal {animal}')
        for merge_out_name, ch_imgs_dict in resized_image_merge_dict.items():
            resized_merge_out_path = os.path.join(resized_merge_out_dir, merge_out_name)
            large_merge_out_path = os.path.join(large_merge_out_dir, merge_out_name)

            for ch_i, img_path_dict in ch_imgs_dict.items():
                og_img_path = img_path_dict['old_path']
                print(f'\t\tprocessing image {og_img_path}')
                img = skimage.io.imread(og_img_path)
                
                # resize
                resized = skimage.transform.resize(img, [int(el/int(RESIZE*100)) for el in img.shape],
                                                anti_aliasing=True)
                # convert resized to png
                resized_norm = normalize8(resized)
                resized_stacked = np.stack((resized_norm,)*3, axis=-1)
                resized_as_pngI = Image.fromarray(resized_stacked)
                enhancer = ImageEnhance.Contrast(resized_as_pngI)
                enhanced_output = enhancer.enhance(2.5)
                # convert large to png
                img_norm = normalize8(img)
                img_as_png = np.stack((img_norm,)*3, axis=-1)
                img_as_pngI = Image.fromarray(img_as_png)
                
                # init merge
                if ch_i == 0:
                    large_img_base = np.stack((np.zeros_like(img_norm),)*3, axis=-1)
                    resized_img_base = np.stack((np.zeros_like(resized_norm),)*3, axis=-1)
                
                # build merged
                large_img_base[...,ch_i] = np.asarray(img_as_pngI)[...,ch_i]
                resized_img_base[...,ch_i] = np.asarray(enhanced_output)[...,ch_i]

                # save chanels
                if not os.path.exists(img_path_dict['large_out_path']):
                    img_as_pngI.save(img_path_dict['large_out_path'])
                if not os.path.exists(img_path_dict['resized_out_path']):    
                    enhanced_output.save(img_path_dict['resized_out_path'])
            
            # save merged
            print(f'\t\tprocessing merged image {og_img_path}')
            large_image = Image.fromarray(large_img_base, mode='RGB')
            large_image.save(large_merge_out_path)
            resized_image = Image.fromarray(resized_img_base, mode='RGB')
            resized_image.save(resized_merge_out_path)





        








    











