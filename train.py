import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub
import datetime
import random
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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
    img = tf.io.decode_jpeg(raw, channels=1)
    # img = tf.image.convert_image_dtype(img, tf.float32) # convert_image_dtype会对图片进行归一化操作，不能除255
    img = tf.image.resize(img, [324, 244])
    img = tf.cast(img, tf.float32)
    img /= 255.0
    # img = (img - 0.5)*2 

    return img


def to_one_hot(image_label):
    return tf.one_hot(image_label, len(class_names))


def make_model(num_classes):
    # m = tf.keras.Sequential([
    #     hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
    #     output_shape=[1280],
    #     trainable=False),  # Can be True, see below.
    # tf.keras.layers.Dense(num_classes, activation='softmax')
    # ])
    # m.build([None, 224, 224, 3])  # Batch input shape.
    # return m

    return tf.keras.Sequential([
        tf.keras.Input((324, 244, 1)),
        tf.keras.layers.experimental.preprocessing.Resizing(
            224, 224, interpolation="bilinear"
        ),
        Conv2D(32, (3, 3), strides=(2, 2), activation="relu", input_shape=(224, 224, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
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
    
    # 此处打乱提高低负载设备性能
    random.seed(123)
    random.shuffle(image_paths)
    random.seed(123)
    random.shuffle(image_labels)
        
    return image_paths, image_labels


def generate_split_dataset(image_paths, image_labels, split_rate=0.8):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(to_one_hot)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    # dataset = dataset.shuffle(dataset.cardinality().numpy(), reshuffle_each_iteration=False) # reshuffle_each_iteration默认为True,每次迭代数据都会重新洗牌

    # 划分数据集
    data_len = dataset.cardinality().numpy()
    train_len = int(data_len*split_rate)
    val_len = data_len - train_len
    
    train_dataset = dataset.take(train_len)
    val_dataset = dataset.skip(train_len).take(val_len)


    print(f"\nloaded {data_len} images.")
    print(f"training on {train_len} images, validating on {val_len} images.\n")

    return train_dataset, val_dataset


def train(train_ds, val_ds, EPOCHS, BATCH_SIZE=32):
    train_ds = train_ds.shuffle(train_ds.cardinality().numpy()).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # 配置tensorboard
    log_dir="runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = make_model(num_classes=len(class_names))
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(
        train_ds,
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=[tensorboard_callback]
        )
    
    model.save('model/model.h5')

    

if __name__ == '__main__':

    data_root = '/home/taozhi/archive/train' # 训练数据根目录
    print(data_root)
    class_names = os.listdir(data_root)
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root)

    train_ds, val_ds = generate_split_dataset(image_paths, image_labels, split_rate=0.7)

    train(train_ds, val_ds, EPOCHS=10, BATCH_SIZE=32)

    