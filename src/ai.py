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
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
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
df_pizza['source_vec'] = df_pizza['source_id'].map(lambda x: to_categorical(x, len(source_enc.classes_)))
print(df['source_vec'].shape)
a = df.iloc[76000].source_vec
df_pizza['source_vec'] = df_pizza['source_vec'].map(lambda x: a)
# print(df.iloc[76000].source_vec)
# print(df_pizza.head())
df_concat = [df, df_pizza]
df_all = pd.concat(df_concat, ignore_index=True)
df_all['is_pizza'].hist(figsize = (20, 7), xrot = 90)
plt.show()

print(df_all['is_pizza'].value_counts())
test_df, train_df = train_test_split(df_all, test_size=0.3, random_state=1, stratify=df_all[['source_id']])
print(train_df.is_pizza.value_counts())
print(test_df.is_pizza.value_counts())
print(df_all)