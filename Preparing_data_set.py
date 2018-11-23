import PIL.Image
import pandas as pd
import numpy as np
import pickle
from scipy import misc
import matplotlib.pyplot as plt

PATH_DATASET = ''

def preprocess():
    """
    Saving the data extracted from images
    """
    data = pd.read_csv(PATH_DATASET + '\out_test.csv', delimiter=';')
    all_images = list()
    all_label = list()
    for path, label in zip(data.Path, data.Class):
        print(path[1:], ':', label)
        img = PIL.Image.open(path[1:])
        img = np.array(img)
        # print(img.shape)
        img = misc.imresize(img, (150, 200, 3), interp='cubic')
        all_images.append(img)
        all_label.append(label)
    img_shape = all_images[0].shape
    img_data = np.array(all_images).reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
    img_label = np.array(all_label).reshape((-1, 1))
    all_data = [img_data, img_label]

    with open(PATH_DATASET + '\out_test.pickle', "wb") as bt:
        pickle.dump(all_data, bt)
        print('Model saved!')

def read_data():
    with open(PATH_DATASET + '/out_test.pickle', 'rb') as bt:
        my_data = pickle.load(bt)

    plt.imshow(my_data[0][0])
    plt.show()
    return my_data

if __name__ == '__main__':
    preprocess()
    read_data()
