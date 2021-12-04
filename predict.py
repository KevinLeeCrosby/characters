#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import load_model

color_mode="grayscale"
image_shape=(128, 128, 1)
batch_size=32

model = load_model("characters2.h5")


def load_file_list():
    classesDF=pd.read_csv("classlabels.csv")
    trnDF = pd.read_csv("TRN.csv").merge(classesDF, on="label").drop("label", axis=1)
    tstDF = pd.read_csv("TST.csv").merge(classesDF, on="label").drop("label", axis=1)
    valDF = pd.read_csv("VAL.csv").merge(classesDF, on="label").drop("label", axis=1)
    df = pd.read_csv("ALL.csv").merge(classesDF, on="label").drop("label", axis=1)
    #df = pd.concat([trnDF, tstDF, valDF], axis=0)
    #df = df[df["name"].str.startswith("GoodImg")].drop_duplicates().sort_values(["class", "name"])
    return df, valDF, tstDF


def preprocessing_function(img):
    """Input and output same shape of image"""
    depth = np.ndim(img)
    is_grayscale = depth == 2
    if depth == 3:
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1) # remove singleton dimensions
            is_grayscale = True
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    img = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=3, sigmaY=3).astype('uint8') # Gaussian smoothing
    #img = cv2.medianBlur(img, ksize=5).astype('uint8') # median smoothing
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU) # Otsu's thresholding
    img = cv2.Canny(img, 100, 200) # edge detector
    if depth == 3:
        img = np.expand_dims(img, axis=2) # 2D -> 3D
        if not is_grayscale:
            img = np.repeat(img, 3, axis=2) # treat grayscale as color
    return img.astype('float64')


def make_image_generator(df, x_col='name', y_col='class', directory=os.getcwd(), image_shape=image_shape, \
                         color_mode=color_mode, batch_size=batch_size, shuffle=False):
    image_generator = ImageDataGenerator(rescale=1/255, preprocessing_function=preprocessing_function)
    return image_generator.flow_from_dataframe(df, x_col=x_col, y_col=y_col, directory=directory,
                                               target_size=image_shape[:2], color_mode=color_mode, interpolation='bicubic',
                                               class_mode='categorical', batch_size=batch_size, shuffle=shuffle)


def predict_class(img_gen):
    pred_probabilities = model.predict(img_gen)
    prediction = np.argmax(pred_probabilities, axis=1)
    classes = list(img_gen.class_indices.keys())
    expected = [ classes[i] for i in img_gen.classes ]
    actual = [ classes[p] for p in prediction ]
    return expected, actual


def predict_label(img_gen):
    pred_probabilities = model.predict(img_gen)
    prediction = np.argmax(pred_probabilities, axis=1)
    expected = [ i + 1 for i in img_gen.classes ]
    actual = [ p + 1 for p in prediction ]
    return expected, actual


def write_endpoint_csv(inDF, expected, actual):
    df = inDF.copy()
    df['expected'] = expected
    df['actual'] = actual
    df.drop(['class', 'thumbnail'], axis=1, inplace=True)
    df.to_csv("endpoint2.csv", index=False)


def write_testpred_csv(inDF, gold, system):
    df = inDF.copy()
    df['gold'] = gold
    df['system'] = system
    df.drop('class', axis=1, inplace=True)
    df.to_csv("test_predictions2.csv", index=False)


def main():
    df, valDF, testDF = load_file_list()

    img_gen = make_image_generator(df)
    expected, actual = predict_label(img_gen)
    write_endpoint_csv(df, expected, actual)

    val_img_gen = make_image_generator(valDF)
    needed, predicted = predict_class(val_img_gen)
    print(classification_report(needed, predicted))

    test_img_gen = make_image_generator(testDF)
    gold, system = predict_label(test_img_gen)
    write_testpred_csv(testDF, gold, system)
    print(classification_report(gold, system))


if __name__ == "__main__":
    main()
