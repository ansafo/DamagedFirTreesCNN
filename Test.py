import numpy as np
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH_DATASET = ''

def read_data():
    with open(PATH_DATASET + '/out_test.pickle', 'rb') as bt:
        my_data = pickle.load(bt)
    return my_data

def convert_to_one_hot(labels):
    he = OneHotEncoder()
    one_hot = he.fit_transform(labels, 4)
    return one_hot.toarray()

my_data = read_data()
label = convert_to_one_hot(my_data[1])

model = load_model('trained_model.h5')

y_predict = model.predict(x=my_data[0], verbose=0)
y_predict_num = np.argmax(y_predict, axis=1)
y_true = np.argmax(label, axis=1)

print(np.round(y_predict, 3))
print('Predict:')
print(y_predict_num)
print('True:')
print(y_true)

cls_names = ['1', '2', '3', '4']

from sklearn.metrics import confusion_matrix

cm = []
cm = confusion_matrix(y_true, y_predict_num)

plt.clf()
plt.imshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
classNames = ['Category 1','Category 2', 'Category 3','Category 4']
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
plt.show()
print(cm)

if __name__ == '__main__':
    read_data()


