import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization


def is_image(filename, verbose=False):
    data = open(filename,'rb').read(10)
    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True
    return False


def load_image(image_path):
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img /= 255.0

    return img


def to_one_hot(label_path):

    return tf.one_hot(label_path, 2)


def make_model(num_classes):
    return keras.Sequential([
        Conv2D(32, (5, 5), activation="relu", input_shape=(224, 224, 3)),
        MaxPool2D(2, 2),
        BatchNormalization(),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPool2D(2, 2),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])


def genearte_image_list(data_root):
    
    class_dirs = [os.path.join(data_root, class_name) for class_name in class_names]
    
    image_paths = []
    image_labels = []
    for class_dir in class_dirs:
        files = os.listdir(class_dir)
        for file in files:
            # 过滤不是.jpg后缀文件
            if file.endswith('.jpg'):
                # 过滤不是JPEG格式的图片
                if is_image(os.path.join(class_dir, file), verbose=False) == False:
                    os.remove(os.path.join(class_dir, file))
                    continue

                image_path = os.path.join(class_dir, file)
                image_label = class_names.index(class_dir[len(data_root)+1:])
                image_paths.append(image_path)
                image_labels.append(image_label)
        
    return image_paths, image_labels


if __name__ == '__main__':
    data_root = '/home/taozhi/archive/train'
    print(data_root)
    class_names = os.listdir(data_root)
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(to_one_hot)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    
    data_len = dataset.cardinality().numpy()
    dataset = dataset.shuffle(data_len)
    train_len = int(data_len*0.7)
    val_len = data_len - train_len
    
    print(f"loaded {data_len} pictures.")
    print(f"training on {train_len} images, validatiing on {val_len} images.")
    
    train_dataset = dataset.take(train_len)
    val_dataset = dataset.skip(train_len).take(val_len)

    model = make_model(num_classes=2)
    model.summary()
    model.compile(
        loss='categorical_crossentropy', # one-hot 编码使用categorical_crossentropy损失函数，否则使用sparse_categorical_crossentropy
        optimizer='adam',
        metrics=['accuracy']
    )

    train_dataset = train_dataset.shuffle(train_len).batch(batch_size=32)
    model.fit(train_dataset, epochs=10)