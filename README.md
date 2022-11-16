# preprocessing-QUINT-workflow
Set of interactive image processing tools to help preprocess and organize brain sections for QUINT workflow. 

**Motivation**

	Working with large image files is time-consuming, especially when repetitive steps are required as in image preprocessing. 
	This repo was developed to facilitate preparing tissue section images for aligning them to a brain atlas in software such as QuickNII. 
	Here preprocessing is done on resized images to reduce computation time. 
	Then all changes to resized images are propogated back to the large image which will ultimatly be used for quantification (e.g. in Ilastik). 



**Pipeline**
	
	1) VSI_convert\process_VSI.py Convert large VSI images to .png, threshold, resize, and split channels, using resized images to align to brain atlas
	2) interactive_order_sections.py -> Organize coronal sections from anterior to posterior
	3) dictate image transforms (rotation, flip, or delete) through xlsx produced when ordering sections
	4) preprocess_quicknii.py -> apply order and transforms to all images (channels, merged, and resized) to generate .XML file required for reading into QuickNII
