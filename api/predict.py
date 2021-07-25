import tensorflow as tf
from tensorflow import keras 
from flask_restful import Resource
from flask import Flask,request,render_template
import sys
import os

class Predict(Resource):
	def post(self):
		model = tf.keras.models.load_model('model.h5')
		img = request.files['img_file']
		filename = img.filename
		img.save(os.path.join("/tmp/", filename))
		img = tf.io.decode_jpeg(tf.io.read_file("/tmp/"+filename),channels=3)
		img = tf.cast(img, tf.float32)
		img = tf.image.resize(img, [224,224])
		img = keras.applications.vgg19.preprocess_input(img)
		img = tf.expand_dims(img, axis=0)
		prediction = model.predict(img)
		print(prediction)
		sys.stdout.flush()
		return prediction.tolist()
