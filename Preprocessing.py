import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils import contours
import pickle
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH_DATASET = ''

frame = cv2.imread("Image_test_1.jpg")

img = cv2.imread('Image_test_1.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations = 16)
thresh = cv2.equalizeHist(thresh)
thresh = cv2.dilate(thresh, None, iterations = 12)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

cand_rect = []
cnts_small_area = []
for count, item in enumerate(cnts):
	rect = cv2.contourArea(item)
	if rect < 265 * 265 and rect > 20 * 20:
		(x1, y1, w, h) = cv2.boundingRect(item)
		y2 = y1 + h
		x2 = x1 + w
		# We save the image in a variable
		match_part = frame[y1:y2, x1:x2]
		# cv2.imwrite("match_part" + str(count) + ".jpg", match_part)
		image_path = "match_part" + str(count) + ".jpg"
		image_data = tf.gfile.FastGFile(image_path, 'rb')
		plt.imshow(np.asarray(match_part), cmap=plt.cm.gray)
		plt.show()

	rect_min = cv2.minAreaRect(item)

	rect_area = rect_min[1][0] * rect_min[1][1]
	if rect_area < 255 * 255 and rect_area > 10 * 10:
		cnts_small_area.append(item)
		epsilon = 0.08 * cv2.arcLength(item, True)
		approx = cv2.approxPolyDP(item, epsilon, True)
		box = cv2.boxPoints(rect_min)
		box_d = np.int0(box)
		cv2.drawContours(frame, [box_d], 0, (0, 0, 255), 4)
# cand_rect.append(box)
plt.imshow(np.asarray(frame), cmap=plt.cm.gray)
plt.show()

# Testing the model on new data
PATH_DATASET = ''

def read_data():
    with open(PATH_DATASET + '\out_test.pickle', 'rb') as bt:
        my_data = pickle.load(bt)
    return my_data

def plot_images(images, cls_pred, cls_names):

    fig, axes = plt.subplots(6, 9)
    fig.subplots_adjust(hspace=0.01, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        xlabel = 'Pred: {0}'.format(cls_names[cls_pred[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Test images', fontsize=16)
    plt.show()

def convert_to_one_hot(labels):
    he = OneHotEncoder()
    one_hot = he.fit_transform(labels, 4)
    return one_hot.toarray()

my_data = read_data()
label = convert_to_one_hot(my_data[1])

model = load_model('trained_model.h5')

y_predict = model.predict(x=my_data[0])
y_predict_num = np.argmax(y_predict, axis=1)
y_true = np.argmax(label, axis=1)

cls_names = ['1', '2', '3', '4']
plot_images(my_data[0], cls_pred=y_predict_num, cls_names=cls_names)

# Plot

fig, axes = plt.subplots(ncols=4, figsize=(10, 4))
ax = axes.ravel()

ax[0].imshow(np.asarray(frame), cmap=plt.cm.gray)
ax[0].set_title('RGB image to \nPAN image')

ax[1].imshow(np.asarray(gray), cmap=plt.cm.gray)
ax[1].set_title('PAN image to\ngray threshold image')

ax[2].imshow(np.asarray(blurred), cmap=plt.cm.gray)
ax[2].set_title('Gray image to\nblurred image')

ax[3].imshow(np.asarray(thresh), cmap=plt.cm.gray)
ax[3].set_title('Blurred image\nto contour threshold\ncandidate patches')

for a in ax:
    a.axis('off')

plt.show()

if __name__ == '__main__':
    read_data()



