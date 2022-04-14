import tensorflow as tf


model = tf.keras.models.load_model("model/mnist.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save model to disk
open("model/mnist.tflite", "wb").write(tflite_model)



converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
aquant_model = converter.convert()

open("model/mnist_aquant.tflite", "wb").write(aquant_model)
