import keras_cv
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds

augmenter = keras_cv.layers.Augmenter(
  layers = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0,255)),
    keras_cv.layers.CutMix(),
    keras_cv.layers.MixUp(),
  ]
)

def augment_data(images, labels):
  labels = tf.one_hot(labels, 3)
  inputs = {"images": images, "labels": labels}
  outputs = augmenter(inputs)
  return outputs["images"], outputs["labels"]

dataset = tfds.load('rock_paper_scissors', as_supervised=True, split='train')
dataset = dataset.batch(64)
dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

densenet = keras_cv.models.DenseNet121(include_rescaling= True, include_top = True, classes = 3)

densenet.compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = ['accuracy']
)

densenet.fit(dataset)
