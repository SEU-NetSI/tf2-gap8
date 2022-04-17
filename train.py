import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

import os


def load_image(image_path):
    """
    from a path load one images and convert to numpy ndarray.
    """
    img = Image.open(image_path)
    img = img.convert(mode='RGB')
    img = img.resize((224, 224))
    # plt.imshow(img)
    img = np.array(img)
    img = img / 255.0
    return img


def generate_images_labels(img_root, class_names):
    """
    generate images and labels numpy ndarray from img_root dir.
    """
    image_paths = []
    image_datas = []
    image_labels = []
    class_dirs = [os.path.join(img_root, class_name) for class_name in class_names]
    for class_dir in class_dirs:
        file_names = os.listdir(class_dir)
        file_paths = [os.path.join(class_dir, file_name) for file_name in file_names]
        
        for file_path in file_paths:
            if str(file_path).endswith('.jpg'):
                image_paths.append(file_path)
                label = class_names.index(class_dir[len(img_root)+1:])
                image_labels.append(label)
    
    for image_path in image_paths:
        img = load_image(image_path)
        image_datas.append(img)
        image_labels = image_labels

    image_datas = np.stack(image_datas)
    image_labels = np.stack(image_labels)

    return image_datas, image_labels


def split_train_val_test(image_datas, image_labels, trainset_rate=0.8):
    """
    generate dataset and split dataset in rate[0.8:0.2]
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_datas, image_labels))
    data_len = image_datas.shape[0]
    dataset = dataset.shuffle(data_len)
    
    train_size = round(data_len*trainset_rate)
    val_size = data_len - train_size

    train_set = dataset.take(train_size)
    val_set = dataset.skip(train_size).take(val_size)
    
    return train_set, val_set, train_size, val_size


def make_model(num_classes):
    """
    construct model
    """
    return tf.keras.Sequential([
        Conv2D(32, (5, 5), activation="relu", input_shape=(224,224,3)),
        MaxPool2D(2, 2),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    

def train(train_set, val_set, batch_size, epochs):
    model = make_model(num_classes=2)
    model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'],
        optimizer="adam"
    )
    model.fit(train_set, epochs=epochs, batch_size=batch_size)
    model.evaluate(val_set)


if __name__ == '__main__':

    img_root = '/home/taozhi/archive/PetImages'

    class_names = os.listdir(img_root)
    print(class_names)

    image_datas, image_labels = generate_images_labels(img_root, class_names)
    train_set, val_set, train_size, val_size = split_train_val_test(image_datas, image_labels)
    
    print(f"training on {train_size} images, validate on {val_size} images.")
    
    train_set = train_set.shuffle(train_size).batch(32)

    for ele in train_set.as_numpy_iterator():
        print(ele)

    val_set = val_set.shuffle(val_size).batch(1)
    train(train_set, val_set, batch_size=32, epochs=10)
    

