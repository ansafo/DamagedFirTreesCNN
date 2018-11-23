import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Input, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import DenseNet169
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

PATH_DATASET = ''

def read_data():
    with open(PATH_DATASET + 'out_whith_aug.pickle', 'rb') as bt:
        bt.seek(0)
        my_data = pickle.load(bt)
    return my_data

def plot_images(images, cls_true, cls_pred, cls_names):

    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i])
        xlabel = 'True: {0}, Pred: {1}'.format(cls_names[cls_true[i]], cls_names[cls_pred[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Validation images', fontsize=16)
    plt.show()

cnn_model = VGG19(weights='imagenet', include_top=False, input_shape = (150, 200, 3))
x = cnn_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

m = Model(inputs = cnn_model.input, outputs = predictions)
m.compile(optimizer=Adam(lr=0.0001), loss=['categorical_crossentropy'], metrics=['accuracy'])

def run_evaluation(x, y, test_data, test_valid=None):

    history = m.fit(x=x, y=y, epochs=10, validation_data=test_data, verbose=2)

    # save model to file of HDF5 format
    m.save('trained_VGG19.h5')
    print("Finish!")

    # load trained model from file
    model = load_model('trained_VGG19.h5')
    accuracy = model.evaluate(x=test_valid[0], y=test_valid[1])
    y_predict = model.predict(x=test_valid[0])
    y_predict_num = np.argmax(y_predict, axis=1)
    y_true = np.argmax(test_valid[1], axis=1)

    print(accuracy)
    # print(y_predict)
    print('Predict:')
    print(y_predict_num)
    print(y_true)
    #
    cls_names = ['1', '2', '3', '4']
    plot_images(test_valid[0], cls_true=y_true, cls_pred=y_predict_num, cls_names=cls_names)

    # plot learning curves

    plt.plot(history.history['acc'], label='Train accuracy')
    plt.plot(history.history['val_acc'], label='Valid accuracy')
    plt.title('MODEL ACCURACY')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Valid loss')
    plt.title('MODEL LOSS')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def convert_to_one_hot(labels):
    he = OneHotEncoder()
    one_hot = he.fit_transform(labels, 4)
    return one_hot.toarray()


if __name__ == '__main__':
    orig_data = read_data()
    X_train, X_tmp, y_train, y_tmp = train_test_split(orig_data[0], orig_data[1], random_state=34, test_size=0.2)

    a, b, y_a, y_b = train_test_split(X_tmp, y_tmp, random_state=42, test_size=0.5)

    # global acc

    train_hot = convert_to_one_hot(y_train)
    test_hot = convert_to_one_hot(y_a)
    valid_hot = convert_to_one_hot(y_b)

    #
    run_evaluation(X_train, train_hot, (a, test_hot), (b, valid_hot))
