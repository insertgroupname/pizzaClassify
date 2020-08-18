import numpy as np
import os
import tensorflow as tf
import PIL
import pathlib
import sys
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.applications.densenet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.metrics import classification_report, classification, accuracy_score

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = (299, 299)
batch_size = 32
base_dir = os.path.join('..', 'content', '1864_33884_bundle_archive', 'images')
image_paths = glob(os.path.join(base_dir, '*', '*'))
pizza_dir = os.path.join('..', 'content', 'pizzaGANdata', 'pizzaGANdata', 'images')
image_pizza_path = glob(os.path.join(pizza_dir, '*'))
df_pizza = pd.DataFrame(dict(path=image_pizza_path))
df = pd.DataFrame(dict(path=image_paths))
df['food_name'] = df['path'].map(lambda x: x.split('\\')[-2].replace('_', ' ').strip())
# a = df['path'][0]
# b = a.split('\\')[-1]
df['is_pizza'] = df['food_name'].map(lambda x: 'pizza' if x == 'pizza' else 'not_pizza')
source_enc = LabelEncoder()
df['source_id'] = source_enc.fit_transform(df['is_pizza'])
df['source_vec'] = df['source_id'].map(lambda x: to_categorical(x, len(source_enc.classes_)))
print(df.columns)
df_pizza['food_name'] = 'pizza'
df_pizza['is_pizza'] = 'pizza'
df_pizza['source_id'] = source_enc.fit_transform(df_pizza['is_pizza'])
df_pizza['source_vec'] = df_pizza['source_id'].map(lambda x: to_categorical(x, len(source_enc.classes_), dtype='str'))
print(df['source_vec'].shape)
a = df.iloc[76000].source_vec
df_pizza['source_vec'] = df_pizza['source_vec'].map(lambda x: a)
# print(df.iloc[76000].source_vec)
# print(df_pizza.head())
df_concat = [df, df_pizza]
df_all = pd.concat(df_concat, ignore_index=True)

# Finish prepare data to dataframe
# Start split prepare train and test dataframe
train_df, test_df = train_test_split(df_all, test_size=0.3, random_state=1, stratify=df_all[['source_id']])
print(train_df.info())
print(train_df.head())
print(train_df.iloc[0])
ImageDataGenerator = ImageDataGenerator(samplewise_center=False,
                                        samplewise_std_normalization=False,
                                        horizontal_flip=True,
                                        vertical_flip=False,
                                        height_shift_range=0.1,
                                        width_shift_range=0.1,
                                        rotation_range=5,
                                        shear_range=0.01,
                                        fill_mode='reflect',
                                        zoom_range=0.15)
# use only first 1000 dataset
train_df = train_df[:1000]
test_df = test_df[:1000]
# create image generator
train_gen = ImageDataGenerator.flow_from_dataframe(
    train_df, x_col='path', y_col='is_pizza', class_mode='binary',
    seed=1,
)
test_gen = ImageDataGenerator.flow_from_dataframe(
    test_df, x_col='path', y_col='is_pizza', class_mode='binary',
    seed=1,
)

print('_____________________')
# start tensorflow model create things
num_classes = 2

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(256, 256, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
# end tensorflow model create
checkpoint_path = "../training/cp"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
train_gen.batch_size = 32
epochs = 2
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    callbacks=[cp_callback]
)

predictions = model.predict(test_gen)
# print(test_gen.__len__())
# print(test_gen.class_indices)
# print(predictions)
#
# X, y = next(test_gen)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# ax1.matshow(classification.confusion_matrix(y, predictions))
# ax1.set_title('Training Results')
# vmat = classification.confusion_matrix(y, predictions)
# ax2.matshow(vmat)
# ax2.set_title('Validation Results')
# print('Validation Accuracy: %2.1f%%' % (100 * accuracy_score(y, predictions)))
# print(vmat)
