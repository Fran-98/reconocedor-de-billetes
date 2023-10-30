import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

img_height = 320
img_width = 320

batch_size = 2

model = keras.Sequential([

])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/bill_dataset/',
    labels='inferred',
    label_mode= 'int',
    class_names=[],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width) # resize if not this size,
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='training',
    )