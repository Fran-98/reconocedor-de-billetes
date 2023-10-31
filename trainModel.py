import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

img_height = 320
img_width = 320

batch_size = 2

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(img_width, img_height)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/bill_dataset/',
    labels='inferred',
    label_mode= 'int',
    class_names=[],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height), # resize if not this size,
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='training',
    )

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/bill_dataset/',
    labels='inferred',
    label_mode= 'int',
    class_names=[],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height), # resize if not this size,
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='validation',
    )

def augment(x ,y):
    image = tf.image.resize(x, (img_width, img_height))
    image = tf.image.random_brightness(image, max_delta= 0.05)
    #image = tf.image.flip_left_right(image)
    #image = tf.image.rgb_to_grayscale(image)
    return image, y

ds_train = ds_train.map(augment)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds_train, epochs=10)