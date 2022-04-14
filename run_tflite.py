import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist
import time

def load_img(image_path):
    img = Image.open(image_path)
    # plt.imshow(img)
    # plt.show()
    img = np.array(img, dtype="float32")
    img = np.expand_dims(img, -1)
    print(img.shape)
    return img


# paths = [
#     '/home/taozhi/keras-yolo3/dog.jpg',
#     '/home/taozhi/keras-yolo3/eagle.jpg',
#     '/home/taozhi/keras-yolo3/giraffe.jpg',
#     '/home/taozhi/keras-yolo3/horses.jpg',
#     '/home/taozhi/keras-yolo3/kite.jpg',
#     '/home/taozhi/keras-yolo3/person.jpg']

# x_test = [load_img(path) for path in paths]

# prepare data
(X_Train, Y_Train), (X_Test, Y_Test) = mnist.load_data()
X_Test = X_Test / 255.0
X_Test = tf.expand_dims(X_Test, -1)
x_test = X_Test[:100]
y_test = Y_Test[:100]


# # save gray picture
# count=0
# for i in x_test:
#     image.save_img('samples/' + str(count) + '.pgm', i)
#     count += 1

# # load gray picture
# x_test = []
# for i in range(100):
#     test_img = image.load_img('samples/' + str(i) + '.pgm', grayscale=True)
#     test_img = np.array(test_img)
#     test_img = np.expand_dims(test_img, -1)
#     x_test.append(test_img)

# x_test = np.array(x_test)
# print(x_test.shape)


# load tflite model
model = tf.lite.Interpreter('model/mnist_aquant.tflite')


start = time.time()
# allocate memory for models
model.allocate_tensors()

# get index for input and output tensors
model_input_index = model.get_input_details()[0]["index"]
model_output_index = model.get_output_details()[0]["index"]

# create array to store the results
model_predictions = []

# run interpreter for value and store the result in array
for x_value in x_test:
    x_value_tensor = tf.convert_to_tensor([x_value], dtype=np.float32)
    model.set_tensor(model_input_index, x_value_tensor)
    model.invoke()
    logits = model.get_tensor(model_output_index)
    model_predictions.append(np.argmax(logits))

end = time.time()
# use time
print(f"reference time: {end-start}")

# accuracy
tot = np.sum(np.equal(y_test, model_predictions))
tflite_accuracy = tot / y_test.shape[0]
print(f"tflite accuracy: {tflite_accuracy}")

