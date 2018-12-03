import PIL.Image
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Input, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH_DATASET = ''

def read_data():
    my_data = pickle.Unpickler(PATH_DATASET+'/out_train.pickle').load()
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

def cnn_model(_x, learning_rate):
    _inputs = Input(shape=_x.shape[1:])

    x = Conv2D(filters=96, kernel_size=3, padding='same')(_inputs)
    x = MaxPool2D(pool_size=2, strides=(2, 2), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=(2, 2), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=512, kernel_size=5, padding='same', activation='relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(400, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)
    m = Model(_inputs, x)
    optimizer = Adam(lr=learning_rate)
    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def run_evaluation(x, y, test_data, test_valid=None):
    learning_rate = 0.0001

    model = cnn_model(x, learning_rate)
    history = model.fit(x=x, y=y, epochs=25, validation_data=test_data, verbose=2)

    # save model to file of HDF5 format
    model.save('trained_model.h5')
    print('Save finish!')

    # load trained model from file
    model = load_model('trained_model.h5')

    accuracy = model.evaluate(x=test_valid[0], y=test_valid[1])

    y_predict = model.predict(x=test_valid[0])
    y_predict_num = np.argmax(y_predict, axis=1)
    y_true = np.argmax(test_valid[1], axis=1)

    print(accuracy)
    print(y_predict)
    print('Predict:')
    print(y_predict_num)
    print(y_true)

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

    run_evaluation(X_train, train_hot, (a, test_hot), (b, valid_hot))
