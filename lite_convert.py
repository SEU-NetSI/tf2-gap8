import tensorflow as tf
import tensorflow_hub as hub


model = tf.keras.models.load_model("model/model.h5", custom_objects={'KerasLayer': hub.KerasLayer})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save model to disk
open("model/model.tflite", "wb").write(tflite_model)



# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# aquant_model = converter.convert()

# open("model/mnist_aquant.tflite", "wb").write(aquant_model)
