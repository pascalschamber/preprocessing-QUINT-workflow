# preprocessing-QUINT-workflow
Interactive image processing tools to help preprocess and organize brain sections for QUINT workflow. 


**Motivation**

Quantification of whole brain section requires large, high-resolution images. Working with large image files can be computationally expensive and time consuming, so typically all preprocessing is done on resized images. However it can be difficult keeping track of image transforms and adjustments and propogating these changes to the original images and the specific channels used to derive quantification.
This project was developed to facilitate propogating the adjustments done on resized images back to the original images for aligning them to a brain atlas in the QUINT workflow, and contains some interactive tools to facilitate section ordering. 



**Workflow**
	
1) VSI_convert\process_VSI.py

	Convert large VSI images to .png
2) preprocess_quicknii\merge_and_resize.py

	Threshold, resize, and split channels
2) preprocess_quicknii\interactive_order_sections.py

	Organize coronal sections from anterior to posterior
	Dictate image transforms (rotation, flip, or delete) through xlsx produced when ordering sections
3) preprocess_quicknii\preprocess_quicknii.py

	apply order and transforms to all images (channels, merged, and resized) to generate .XML file required for reading into QuickNII

<img src="workflowDiagram.png" title="Workflow Diagram">


**Usage**
	
	from VSI_convert.VSI_convert import convert_vsi
	from preprocess_quicknii.merge_and_resize import merge_and_resize
	from preprocess_quicknii.interactive_order_sections import interactive_order_sections
	from preprocess_quicknii.preprocess_quicknii import QuickNII_preprocess
	
	convert_vsi(
		base_dir, 
		out_dir_base, 
		series_index, # 0 is highest resolution image
		FILE_TYPE='.vsi',
		DIR_FILTER_STR='',  # exclude directories with this string
		image_fn_skip_string='', # exclude image files
		thresh_dict={
			0:[0, 20000],
			1:[0, 20000],
			2:[0, 20000],
		},
		SEPARATE_CHANNELS=True,
		SAVE_SEPARATE_CHANNELS=True,
		SAVE_IMAGE=True,
		PLOT=True,
		PLOT_SEPARATE=False,
	)
	
	
	merge_and_resize(
	    base_dir, 
	    output_dir, 
	    channel_names_list, # define channel names
	    channel_order_list, # define channel indices
	    RESIZE=0.1, # 10%
	    base_dir_filter='' # exclude directories with this string
    	)
	
	
	# drag and place images on screen to define order
	interactive_order_sections(base_dir, animal_number, channel)
	
	
	'''
	Once all orders have been defined run below to apply image order and image transforms, and propogate these alterations back to all other large/resized images and channels
	'''
	
	QuickNII_preprocess(
		base_dir, 
		base_children_dirs, # e.g. ['resized_images', 'large_images']
		image_order_dir_index, # define directory where alterations were made, e.g. 0
		APPLY_ORDER = True,
		APPLY_TRANSFORMATIONS = True
    	)
	
	
