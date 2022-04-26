import tensorflow as tf
import numpy as np
import time
import os
import random
from keras.preprocessing.image import image
from train import load_image


def load_data(data_root, split_rate=0.7):
    
    class_names = os.listdir(data_root)
    class_dirs = [os.path.join(data_root, class_name) for class_name in class_names]
    
    image_paths = []
    image_labels = []
    image_datas = []
    for class_dir in class_dirs:
        files = os.listdir(class_dir)
        for file in files:
            # 过滤不是.jpg后缀文件
            if file.endswith('.jpg'):
                image_path = os.path.join(class_dir, file)
                image_label = class_names.index(class_dir[len(data_root)+1:])
                image_paths.append(image_path)
                image_labels.append(image_label)
    
    # 此处打乱提高低负载设备性能
    random.seed(123)
    random.shuffle(image_paths)
    random.seed(123)
    random.shuffle(image_labels)
        
    for path in image_paths:
        img = load_image(path)
        image_datas.append(img)
    
    X_train = image_datas[:int(len(image_datas)*split_rate)]
    Y_train = image_labels[:int(len(image_labels)*split_rate)]
    X_test = image_datas[int(len(image_datas)*split_rate):]
    Y_test = image_labels[int(len(image_labels)*split_rate):]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test


def evaluate_tflite(x_test, y_test):
    print(x_test.shape)
    print(y_test.shape)
    print('evaluating tflite accuracy.')
    
    # load model
    model = tf.lite.Interpreter('model/model.tflite')

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


def save_samples(x_test, y_test, len):
    print(f"generate {len} samples for quantize.")
    x_quant = x_test[:len]
    y_quant = y_test[:len]
    count=0
    for i in x_quant:
        image.save_img('samples/' + str(count) + '_' + str(y_quant[count]) + '.pgm', i)
        count += 1

    print("saved samples in samples/")




if __name__ == '__main__':

    data_root = '/home/taozhi/archive/train' # 训练数据根目录

    # prepare data
    X_Train, Y_Train, X_Test, Y_Test = load_data(data_root, split_rate=0.7)
    
    # run tflite
    evaluate_tflite(X_Test, Y_Test)
    
    # save picture for quantize
    save_samples(X_Test, Y_Test, len=50)    

