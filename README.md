# Deep-Computational-AO
Computational adaptive optics via deep learning algorithms

The matlab part is used to load the aberration, and the python part is used for deep learning.
## Citation Statement
The matlab part refers to https://github.com/eguomin/DeAbePlus/

## Project Introduction
The entire workflow is roughly divided into two steps: the first step is to prepare the training set, and the second step is to perform training.

### Prepare data
The original image after denoise can be loaded with the disparity as a training set through the tif_blurred.m.
Then use the get_data.m to make training file which can be identification by the dataloader.

