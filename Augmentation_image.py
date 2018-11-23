import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage import io
import numpy as np
import os as os
import PIL.Image as Image

def transform_img(img_file_name_full, Folder_name, img_file_name):

    Exception="Image.jpg"

    def multiply_image(image,R,G,B):
        image=image*[R, G, B]
        cv2.imwrite(Folder_name + img_file_name + "Multiply-"+str(R)+str(G)+str(B)+Exception, image)

    def gausian_blur(image, blur):
        image=cv2.GaussianBlur(image,(5,5), blur)
        cv2.imwrite(Folder_name + img_file_name + "GaussianBlur-"+str(blur)+Exception, image)

    def averageing_blur(image, shift):
        image=cv2.blur(image,(shift, shift))
        cv2.imwrite(Folder_name + img_file_name + "AverageingBlur-"+str(shift)+Exception, image)

    def rotate_image(image, angle):
        rows, cols, pixs = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
        cv2.imwrite(Folder_name + img_file_name + "Rotation-" + str(angle) + Exception, image)

    def crop_image(image, size_h, size_w):
        size_h_ = round(size_h/2)
        size_w_ = round(size_w/2)
        x, y, _ = image.shape
        center_x = round(x/2)
        center_y = round(y/2)
        image = image[center_x-size_h_:center_x+size_h_, center_y-size_w_:center_y+size_w_, :]
        cv2.imwrite(Folder_name + img_file_name + "Crop-" + str(size_h) + 'x' +str(size_w) + Exception, image)

    image = cv2.imread(img_file_name_full)
    crop_image(image, 40,40)
    multiply_image(image,1.25,1,1)
    multiply_image(image,1.5,1.5,1.5)
    gausian_blur(image, 0.50)
    averageing_blur(image, 4)
    rotate_image(image, 0)
    rotate_image(image, 5)
    rotate_image(image, 20)
    rotate_image(image, 25)
    rotate_image(image, 45)
    rotate_image(image, 50)
    rotate_image(image, 90)
    rotate_image(image, 180)

open_directory = 'Class/1'
save_directory = 'Class_preprocessing'
files = [f for f in os.listdir(open_directory)]
for img_file_name in files:
    img_file_name_full = open_directory + '/' + img_file_name
    transform_img(img_file_name_full, save_directory, img_file_name)
