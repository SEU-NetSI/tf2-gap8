import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.datasets import fashion_mnist, mnist

def make_model(num_classes):

    return tf.keras.models.Sequential([
        Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(rate=0.5),
        Dense(num_classes, activation="softmax"),
    ])



num_classes = 10
model = make_model(num_classes)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=['accuracy'],
)
model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
print(x_train.shape)
print(y_train.shape)

history = model.fit(x_train, y_train, epochs=3, batch_size=32)
model.save("model/mnist.h5")




