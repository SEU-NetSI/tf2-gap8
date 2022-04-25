import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub
import datetime


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
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
        trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    m.build([None, 224, 224, 3])  # Batch input shape.
    return m


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

    data_root = '/home/taozhi/archive/train' # 训练数据根目录
    print(data_root)
    class_names = os.listdir(data_root)
    print(class_names)

    image_paths, image_labels = genearte_image_list(data_root)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(load_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(image_labels).map(to_one_hot)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    dataset = dataset.shuffle(dataset.cardinality().numpy())

    # 划分数据集
    data_len = dataset.cardinality().numpy()
    train_len = int(data_len*0.7)
    val_len = data_len - train_len
    
    train_dataset = dataset.take(train_len)
    val_dataset = dataset.skip(train_len).take(val_len)

    model = make_model(num_classes=len(class_names))
    model.summary()

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    log_dir="runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print(f"loaded {data_len} images.")
    print(f"training on {train_len} images, validating on {val_len} images.")

    train_dataset = train_dataset.shuffle(train_len).batch(batch_size=32)
    val_dataset = val_dataset.batch(32)

    model.fit(
        train_dataset,
        validation_data=val_dataset, 
        epochs=20, 
        callbacks=[tensorboard_callback]
        )
    
    model.save('model/resnet.h5')