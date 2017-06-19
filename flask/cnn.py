from mxnet_predict import Predictor
import numpy as np
import cv2
import os

prefix = '../model/ft-resnext-18'
symbol_file = "%s-symbol.json" % prefix
param_file = "%s-0118.params" % prefix
predictor = Predictor(open(symbol_file, "r").read(),
                      open(param_file, "rb").read(),
                      {'data': (1, 3, 224, 224)})

def preprocess(img):
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy:yy + short_egde, xx:xx + short_egde]
    resized_img = cv2.resize(crop_img, (224, 224))
    sample  = resized_img - [123.68, 116.779, 103.939]
    sample = np.transpose(resized_img, (2, 0, 1))
    sample = sample[::-1, :, :]

    return sample

def predict(img_array):
    img = cv2.imdecode(np.fromstring(img_array, np.uint8), 1)
    if img is not None:
        batch = preprocess(img)
        predictor.forward(data=batch)
        output = predictor.get_output(0)
        print output
        prob = output[0][1]
        print prob
        return prob > 0.5
    return False


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    img_array = open(path).read()
    print predict(img_array)

def test(path):
    files = os.listdir(path)
    correct = 0
    total = 0
    image_list = [image for image in files if  not image.startswith('.')]
    for image in image_list:
        img_array = open(os.path.join(path, image)).read()
        total += 1
        if predict(img_array) == 0:
            correct += 1
    print "test image number: %d", total
    print "correct rate: %f", correct*1.0/total
