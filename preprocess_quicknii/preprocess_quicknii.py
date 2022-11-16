import os
from pathlib import Path
import re
import skimage 
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd

import image_processing_utils as u2

def get_dir_contents(adir):
    return [Path(os.path.join(adir, c)) for c in os.listdir(adir)]

def get_animals_to_process(base_dir, base_children_dirs, image_order_dir_index, skip_dirs):
    '''init directories and remove folders to skip'''
    dirs_to_search = [os.path.join(base_dir, sub_dirr) for sub_dirr in base_children_dirs]
    image_order_dir = os.path.join(base_dir, base_children_dirs[image_order_dir_index])
    animals_to_process = skip_animal_dirs(image_order_dir, skip_dirs)
    return animals_to_process, dirs_to_search, image_order_dir

def skip_animal_dirs(animals_dir, animals_to_skip):
    '''remove folders to skip'''
    animals = [el.stem for el in get_dir_contents(animals_dir) if os.path.isdir(el)]
    for skip_dir in animals_to_skip:
        if skip_dir in animals:
            animals.remove(skip_dir)
    return animals

def get_all_images_with_same_name(image_name, animal_id, dirs_to_search):
    '''walk resized and large dirs, collect images with same base name but distributed through other folders'''
    all_images = []
    ddirs_to_search = [os.path.join(adir, animal_id) for adir in dirs_to_search]
    
    for adirr in ddirs_to_search:
        adirr_contents = get_dir_contents(adirr)
        for adirr_content in adirr_contents:
            if not os.path.isdir(adirr_content): # skip non directories
                continue
            sub_adirr_contents = get_dir_contents(adirr_content)
            for sub_adirr_content in sub_adirr_contents:
                if image_name in sub_adirr_content.stem:
                    all_images.append(sub_adirr_content)
    return all_images

def get_image_name(astr):
    ''' get just the 'Image_xx' part from a file string '''
    name_pattern = 'Image_\d+_'

    if isinstance(astr, Path):
        astr = astr.stem

    name_match = re.search(name_pattern, astr)
    if not name_match:
        raise ValueError(astr)
    
    id_ind = name_match.span()[-1]
    img_name = astr[:id_ind]
    return img_name
    

def make_order_ch_agnostic(aseries):
    ''' extract just the image name from the image path stem '''
    to_return = []
    
    for el in aseries:
        img_name = get_image_name(el)
        to_return.append(img_name)
        
    if len(to_return) == 1:
        to_return = to_return[0]
    return to_return

def apply_order (order_df, sub_c):
    
    img_name = make_order_ch_agnostic([sub_c.stem])
    assert isinstance(img_name, str)

    rename_to_sindex = order_df[order_df['to_img_name'] == img_name]['sindex']
    if len(rename_to_sindex) != 1:
        raise ValueError(sub_c.name)
    
    match = re.search('s\d\d\d', sub_c.stem)
    if not match:
        raise ValueError (sub_c.name)
    
    new_stem = sub_c.stem[:match.span()[0]] + rename_to_sindex.values[0]
    new_stem += sub_c.suffix

    return new_stem

def get_order_df(adir):
    ''' create the s_xxx image order '''
    excel_path = os.path.join(adir, 'ImageOrder.xlsx')
    order_df = pd.read_excel(excel_path)
    order_df['sindex'] = ['s'+str((4*i)+1).zfill(3) for i in range(len(order_df['Order']))]
    order_df['to_img_name'] = make_order_ch_agnostic(order_df['Order'].values)

    # verify transforms
    t_cols = ['h_flip', 'rotate', 'remove']
    t_vals = [order_df[acol] for acol in t_cols]
    # TODO

    return order_df      

def get_transforms(order_df, image_name):
    transforms_dict = {'rotate':False, 'h_flip':False, 'remove':False}
    row = order_df[order_df['to_img_name'] == image_name]
    assert len(row) == 1, f'{image_name} returned {len(row)} hits'
    for k in transforms_dict:
        v = row[k].values[0]
        if str(v) != 'nan':
            transforms_dict[k] = v

    # just return False if no transforms to apply
    all_false = True
    for k,v in transforms_dict.items():
        if v != False: 
            all_false = False
    if all_false:
        return False

    return transforms_dict

def apply_transforms(path_to_image, transforms_to_apply):
    if transforms_to_apply['remove'] != False:
        os.remove(path_to_image)
        return

    image = skimage.io.imread(path_to_image, plugin='simpleitk')

    for t_name, t_val in transforms_to_apply.items():
        if t_val:
            if t_name == 'h_flip':
                image = np.fliplr(image)
            elif t_name == 'rotate':
                # convert from 360 degrees to np rot degree ints and make clockwise
                np_degrees = (360-t_val)/90 
                image = np.rot90(image, np_degrees)
    
    skimage.io.imsave(path_to_image, image, check_contrast=False)

def transforms_processing(order_df, image_name, image_paths):
    ''' pipeline for getting transforms specified in ImageOrder and applying to all images if needed '''
    transforms = get_transforms(order_df, image_name)
    if not transforms:
        return

    print(f'\tGathering images for {image_name} to apply transforms')
    for image_path in image_paths[:]:
        print(f'\t\tapplying transforms to {Path(image_path).stem}')
        apply_transforms(image_path, transforms)


def get_packet(sub_c):
    ''' create a dict that will be used to create elements for each image in the XML file '''
    fb_pattern = '_s(\d\d\d)'
    nr_match = re.search(fb_pattern, sub_c.stem)
    if not nr_match:
        raise ValueError(sub_c.stem)
    
    sindex = nr_match.groups(0)[0]
    
    img = u2.read_image(sub_c)

    packet = {
        'filename' : sub_c.name,
        'nr' : str(int(sindex)),
        'width':str(img.shape[1]),
        'height':str(img.shape[0]),
    }
    return packet

def file_builder(packets, sub_directory):
    ''' build the packets into an XML file '''
    fn = 'filebuilderoutput.xml'
    out_path = os.path.join(sub_directory, fn)

    data = ET.Element('series')
    data.set('name', fn[:-4])

    sorted_packets = sorted(packets, key=lambda k: (int(k['nr'])))
    for packet in sorted_packets:
        elem = ET.SubElement(data, 'slice')
        for k,v in packet.items():
            elem.set(k, v)
    
    b_xml = ET.tostring(data)

    with open(out_path, "wb") as f:
        f.write(b_xml)




def QuickNII_preprocess(
        base_dir, 
        base_children_dirs, 
        image_order_dir_index, 
        APPLY_ORDER = True,
        APPLY_TRANSFORMATIONS = True,
        skip_dirs=None, 
        start_animal_str=None, 
        start_image_index=None,
    ):
    '''
    ##########################################################################################
    PARAMS
    ~~~~~~
        base_dir: str: define directory that contains resized and full_images
        base_children_dirs: list of strs: list of sub dirs for above, just relative path needed
        image_order_dir_index: int: define index in children_dirs list that was used to derive image order
        APPLY_ORDER: bool: rewrite filenames to have slide number suffix
        APPLY_TRANSFORMATIONS: bool: apply image transforms described in imageorder xlsx file
        skip_dirs: list of strs: list of folders in image_order_dir to skip
        start_animal_str, start_image_index, : str, int: define which animal and image num to start at (if progress was halted)
    ##########################################################################################
    '''
    
    animals_to_process, dirs_to_search, image_order_dir = get_animals_to_process(
        base_dir, base_children_dirs, image_order_dir_index, skip_dirs)

    for animal_id in animals_to_process:
        
        # get channel folders in an animals directory
        animal_dir = os.path.join(image_order_dir, animal_id)
        dir_contents = get_dir_contents(animal_dir)

        # specify s_xxx image order, after ordering
        order_df = get_order_df(animal_dir)

        print(f'processing animal {animal_id}...')
        # apply transforms
        ##########################################################################################
        if APPLY_TRANSFORMATIONS:
            for sub_dir in dir_contents[:]:
                
                # skip non directories
                if not os.path.isdir(sub_dir): 
                    continue
                # point to this directory to get all images of same name to base transforms on
                if not Path(sub_dir).name == 'Merged': 
                    continue
                
                sub_dir_contents = get_dir_contents(sub_dir)

                # useful if progress was halted to start from a specific point
                content_start=0
                if animal_id == start_animal_str: content_start=start_image_index
                    
                # iterate through image files
                for sub_c_i, sub_c in enumerate(sub_dir_contents[content_start:]):
                    # skip non image files
                    if Path(sub_c).suffix != '.png':
                        print(f'!!! skipping {sub_c.name}')   
                        continue   
                    
                    # get image name
                    image_name = get_image_name(sub_c)
                    # get all images with same name
                    all_image_paths = get_all_images_with_same_name(image_name, animal_id, dirs_to_search)

                    # get transforms and apply if needed
                    transforms_processing(order_df, image_name, all_image_paths)


        # apply image ordering
        ##########################################################################################
        if APPLY_ORDER:
            for sub_dir in dir_contents[:]:
                
                if not os.path.isdir(sub_dir): # skip non directories
                    continue
                print(f'\tapplying order to {sub_dir.stem}')

                sub_dir_contents = get_dir_contents(sub_dir)
                
                packets = [] # store XML elements here

                for sub_c_i, sub_c in enumerate(sub_dir_contents[:]):
                    
                    # skip non image files
                    if Path(sub_c).suffix != '.png': 
                        print(f'\t\t!!! skipping {sub_c.name}')   
                        continue
                    
                    # get renamed s index
                    new_name = apply_order(order_df, sub_c)

                    # rename file 
                    new_path = os.path.join(sub_dir, new_name)
                    os.rename(sub_c, new_path)

                    # file builder 
                    packets.append(get_packet(Path(new_path)))

                # file builder
                file_builder(packets, sub_dir)



'''
#######################################################################################################
MAIN
#######################################################################################################
'''

if __name__ == '__main__':
    # specify which operations to perform
    APPLY_ORDER = bool(1)
    APPLY_TRANSFORMATIONS = bool(1)
    # define directory that contains resized and full images
    base_dir = r'D:\TEL Slides\ScriptOutput2'
    # define dirs that contain images to be ordered and transformed
    base_children_dirs = ['resized_images', 'large_images']
    # define dir that was used to derive image order
    image_order_dir_index = 0 
    # skip some animals, useful if progress was halted
    skip_dirs = ['TEL1', 'TEL14','TEL10', 'TEL11', 'TEL12', 'TEL13', 'TEL2', 'TEL3', 'TEL4', 'TEL5', 'TEL6']
    # define which animal and image num to start at (if progress was halted)
    start_animal_str = ''
    start_image_index = 0

    QuickNII_preprocess(
        base_dir, 
        base_children_dirs, 
        image_order_dir_index, 
        APPLY_ORDER = APPLY_ORDER,
        APPLY_TRANSFORMATIONS = APPLY_TRANSFORMATIONS,
        skip_dirs=skip_dirs, 
        start_animal_str=start_animal_str, 
        start_image_index=start_image_index,
    )



    


        

        
