import tensorflow as tf


model = tf.keras.models.load_model("/home/taozhi/keras-yolo3/model_data/yolo-tiny.h5", compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save model to disk
open("model/yolo-tiny.tflite", "wb").write(tflite_model)



# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# aquant_model = converter.convert()

# open("model/mnist_aquant.tflite", "wb").write(aquant_model)
