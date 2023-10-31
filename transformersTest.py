from transformers import pipeline, AutoImageProcessor, DefaultDataCollator, create_optimizer, TFAutoModelForImageClassification, trainer
from datasets import load_dataset
import evaluate
import numpy as np
import tensorflow as tf
from PIL import Image

import os
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

#dataset_path = 'dataset/bill_dataset/'

#dataset = load_dataset('imagefolder', data_dir=dataset_path, split= 'train')
#dataset.push_to_hub('Franman/billetes-argentinos', token=os.environ['HF_WRITE_TOKEN'])

dataset = load_dataset('Franman/billetes-argentinos', split= 'train')
dataset = dataset.train_test_split(test_size=0.2)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

size = (image_processor.size["height"], image_processor.size["width"])

train_data_augmentation = keras.Sequential(
    [
        tf.keras.layers.RandomCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="train_data_augmentation",
)

val_data_augmentation = keras.Sequential(
    [
        tf.keras.layers.CenterCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)

def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

dataset["train"].set_transform(preprocess_train)
dataset["test"].set_transform(preprocess_val)


data_collator = DefaultDataCollator(return_tensors="tf")

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

batch_size = 100
num_epochs = 10
num_train_steps = len(dataset["train"]) * num_epochs
learning_rate = 0.01
weight_decay_rate = 0.01

optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=0,
)

model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# converting our train dataset to tf.data.Dataset
tf_train_dataset = dataset["train"].to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# converting our test dataset to tf.data.Dataset
tf_eval_dataset = dataset["test"].to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)

push_to_hub_callback = PushToHubCallback(
    output_dir="model/clasificador-de-pesos",
    hub_model_id="Franman/clasificador-de-pesos",
    tokenizer=image_processor,
    save_strategy="no",
)

callbacks = [metric_callback, push_to_hub_callback]

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)



