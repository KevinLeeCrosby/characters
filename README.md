# Capstone - Character Recognition in Natural Images

Capstone Project for Data Science Immersive - Weekend 2021-04

Divergence Academy

September 19, 2021

## Dataset

![chars74k](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/chars74k.jpg)

[Chars74K Dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/index.html)

The labels are 0-9, A-Z, and a-z.

## Objective

The main objectives was to train a deep Computer Vision (CV) Convolutional Neural Network (CNN) with the highest accuracy possible.

This task was chosen to augment my Natural Language Processing (NLP) skills, and possibly bridge the two Machine Learning fields.

## Updated Results - November 14th, 2021

I continued to improve the model by preprocessing the images with Image Processing.

The final results boosted the test accuracy increased to 74%.

The updated Powerpoint and Power BI file will be forthcoming and linked below.

In summary, these were the Image Processing enhancements:

<div class="row">
  <div class="column">
    <img src="https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/D1.png" alt="color" width="128" height="128">
    <img src="https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/D2.png" alt="grayscale" width="128" height="128">
    <img src="https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/D3.png" alt="smoothing" width="128" height="128">
    <img src="https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/D4.png" alt="thresholding" width="128" height="128">
    <img src="https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/D5.png" alt="edge_detection" width="128" height="128">
  </div>
</div>

That is, from the original color image, convert to grayscale, Gaussian smooth, adaptive Otsu thresholding, Canny edge detection.

This is in addition to resizing of the image to 128 by 128 pixels and scaling the pixels to be in the range from 0 to 1.

Also, the training set had random rotations, translations, zoom, and shear as well.

Refer to the following for details:

Powerpoint Presentation: TBD

Power BI Demo: TBD  (includes more examples)

[Jupyter Notebook](https://github.com/KevinLeeCrosby/characters/blob/main/Characters2.ipynb)

## Preliminary Results - September 19th, 2021

After working on the project for the allotted 1 week, the test accuracy of 60% was achieved.

The lower score is partially due to look alike characters, and inexperience building CV CNNs.

The images were converted to grayscale, resized to 128 by 128 pixels, and rescaled with pixel intensities between 0 and 1.

The training set had random rotations, translations, zoom, and shear as well.

Refer to the following for details:

[Powerpoint Presentation](https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/Character%20Recognition%20in%20Natural%20Images.pptx)

[Power BI Demo](https://raw.githubusercontent.com/KevinLeeCrosby/characters/main/Characters.pbix)

[Jupyter Notebook](https://github.com/KevinLeeCrosby/characters/blob/main/Characters.ipynb)
