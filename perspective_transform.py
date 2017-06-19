import numpy as np
import cv2
import os


data_path='./health'
data_output_path = './health/'

image_list = os.listdir(data_path)
image_list = [image for image in image_list if not image.startswith('.')]

for image in image_list:
    image_name = os.path.join(data_path, image)
    img = cv2.imread(image_name)
    if img is None:
        continue
    height, width, color = img.shape
    pts1 = np.float32([[0,0],[width, 0],[0, height],[width, height]])
    pts2 = np.float32([[0,0],[224, 0],[0, 224],[224,224]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (224, 224))
    cv2.imwrite(os.path.join(data_output_path, image+'_output.jpg'), dst)
