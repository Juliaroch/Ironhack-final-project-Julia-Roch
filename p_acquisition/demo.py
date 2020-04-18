#demo

import pandas as pd
import numpy as np
import os
import matplotlib 
import matplotlib.pyplot as plt
import random
import json

import tensorflow as tf
from tensorflow.python.keras.models import model_from_json, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import *

images ='../data_images/resized/resized_demo_test'
data_artists = pd.read_csv('../data/processed/data_artists.csv')
artists_top_name = data_artists['name'].str.replace(' ', '_').values
train_input_shape = (128, 128, 3)
n_classes = artists_top_name.shape[0]

def get_images(images):
    train_datagen=ImageDataGenerator(rescale=1./255.,horizontal_flip=True,vertical_flip=True)
    train_generator=train_datagen.flow_from_directory(directory=images,class_mode='categorical',target_size=train_input_shape,shuffle=True,classes=artists_top_name.tolist())
    return train_generator

model = load_model('../data/results/my_model170420RNv3.h5')

n =10
fig, axes = plt.subplots(1, n, figsize=(25,10))

def demo(train_generator,n):
    for i in range(n):
        random_artist = random.choice(artists_top_name)
        random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
        random_image_file = os.path.join(images_dir, random_artist, random_image)

    # Original image
        test_image = image.load_img(random_image_file, target_size=(train_input_shape))

    # Predict artist
        test_image = image.img_to_array(test_image)
        test_image /= 255.
        test_image = np.expand_dims(test_image, axis=0)

        prediction = model.predict(test_image)
        prediction_probability = np.amax(prediction)
        prediction_idx = np.argmax(prediction)

        labels = train_generator.class_indices
        labels = dict((v,k) for k,v in labels.items())
    
        title = "Actual artist = {}\nPredicted artist = {}\nPrediction probability = {:.2f} %".format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),prediction_probability*100)
        # Print image
        axes[i].imshow(plt.imread(random_image_file))
        axes[i].set_title(title)
        axes[i].axis('off')

        plt.show()

    return plt.savefig('../data/results/images_prediction.pdf')