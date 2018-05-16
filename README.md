# Selfie Filters Using OpenCV
This python application can put various sunglasses on a detected face (I am calling them 'Selfie Filters') by finding the Facial Keypoints (15 unique points). These keypoints mark important areas of the face - the eyes, corners of the mouth, the nose, etc.

## Code Requirements
The code is in Python (version 3.6 or higher). You also need to install OpenCV and Keras libraries.

## Description
OpenCV is often used in practice with other machine learning and deep learning libraries to produce interesting results. Employing **Convolutional Neural Networks (CNN)** in (Keras)[https://keras.io/] along with OpenCV - I built a couple of selfie filters (very boring ones).

Facial keypoints can be used in a variety of machine learning applications from face and emotion recognition to commercial applications like the image filters popularized by Snapchat.

<img src="images/keypoints_test_results.png" width=400 height=300/>

Facial keypoints (also called facial landmarks) are the small blue-green dots shown on each of the faces in the image above - there are 15 keypoints marked in each image.

I used [this dataset from Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data) to train a decent CNN model to predict the facial keypoints given a face and used the keypoints to place the desired filters on the face (as shown below).

## Working Example
<img src="https://github.com/akshaychandra111/Selfie_Filters_OpenCV/blob/master/demo.gif">

## Execution
Order of Execution is as follows:

Step 0 - Download the _'training.zip'_ file from [here](https://www.kaggle.com/c/facial-keypoints-detection/data) and extract it into the _'data'_ folder.

Step 1 - Execute ``` python model_builder.py ```

Step 2 - This could take a while, so feel free to take a break.

Step 3 - Execute ``` python shades.py ```

Step 4 - Choose filters of your choice.

Step 5 - And don't forget to SMILE !
