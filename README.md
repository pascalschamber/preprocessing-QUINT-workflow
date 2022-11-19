# preprocessing-QUINT-workflow
Interactive image processing tools to help preprocess and organize brain sections for QUINT workflow. 


**Motivation**

Quantification of whole brain section requires large, high-resolution images. Working with large image files can be computationally expensive and time consuming, so typically all preprocessing is done on resized images. However it can be difficult keeping track of image transforms and adjustments and propogating these changes to the original images and the specific channels used to derive quantification.
This project was developed to facilitate propogating the adjustments done on resized images back to the original images for aligning them to a brain atlas in the QUINT workflow, though it also contains some interactive tools to facilitate section ordering. 



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