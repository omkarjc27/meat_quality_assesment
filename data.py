import tensorflow as tf
from tensorflow import keras
import pandas as pd

def load_img(path,fresh):
    img = tf.io.decode_jpeg(tf.io.read_file(path),channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [224,224])
    img = keras.applications.vgg19.preprocess_input(img)
    return img,fresh

def load_test_img(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path),channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [224,224])
    img = keras.applications.vgg19.preprocess_input(img)
    return img