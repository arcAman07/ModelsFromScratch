import keras_cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import losses
from tensorflow.keras import optimizers

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()
data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
val_steps_per_epoch = dataset_info.splits["validation"].num_examples // BATCH_SIZE

IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes

def to_dict(image, label):
  image = tf.image.resize(image, IMAGE_SIZE)
  image = tf.cast(image, tf.float32)
  label = tf.one_hot(label, num_classes)
  return {"images": image, "labels": label}

def prepare_dataset(dataset, split):
  if split == "train":
    return (
      dataset.shuffle(10*BATCH_SIZE)
      .map(to_dict, num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
    )
  if split == "test":
    return (
      dataset.map(to_dict, num_parallel_calls=AUTOTUNE)
      .batch(BATCH_SIZE)
    )

def load_dataset(split="train"):
  dataset = data[split]
  return prepare_dataset(dataset, split)

train_dataset = load_dataset()

def visualize_dataset(dataset, title):
  plt.figure(figsize=(6,6)).suptitle(title, fontsize=18)
  for i, samples in enumerate(iter(dataset.take(9))):
    images = samples["images"]
    plt.subplot(3,3,i+1)
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.axis("off")
  plt.show()

# visualize_dataset(train_dataset, "Training Dataset")

  # RAND AUGMENT

rand_augment = keras_cv.layers.RandAugment(
    value_range=(0,255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=0.5,
)

def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs
  
train_dataset = load_dataset().map(apply_rand_augment, num_parallel_calls = AUTOTUNE)

visualize_dataset(train_dataset, title= "After RandAugment")
