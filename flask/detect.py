import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import xgboost as xgb
import tempfile

bins = 255
P = 16
R = 2

bst = xgb.Booster({'nthread': 4})
bst.load_model('../data/xgb.model')

def detector_return(img):
    detector = cv2.CascadeClassifier('../data/cascade.xml')
    bbox = detector.detectMultiScale(img, scaleFactor=1.1,
                                     minNeighbors=25, minSize=(100, 100))
    return bbox

def write_image(bbox, res):
    x = [n[2] * n[3] for n in bbox]
    max_index = x.index(max(x))
    print bbox[max_index]
    x, y, w, h = bbox[max_index]
    return res[y:(y + h), x:(x + w)]

def classify(crop):
    re = cv2.resize(crop, (640, 400))
    file = tempfile.mktemp(dir='static') + '.jpg'
    cv2.imwrite(file, crop)
    cv2.imwrite('re.jpg', re)
    lbp = np.histogram(local_binary_pattern(re, P, R), bins=bins)[0]
    dx = xgb.DMatrix([lbp])
    z = bst.predict(dx)
    print z[0]
    return z[0] >= 0.5, file

def predict(img_array):
    img = cv2.imdecode(np.fromstring(img_array, np.uint8), 0)

    if img is not None:
        resize_max = 720
        h, w = img.shape
        if h > w:
            img = cv2.resize(img, (resize_max * w / h, resize_max), cv2.cv.CV_INTER_CUBIC)
        else:
            img = cv2.resize(img, (resize_max, resize_max * h / w), cv2.cv.CV_INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        res = clahe.apply(img)
        rows, cols = img.shape
        res = cv2.copyMakeBorder(res, 100, 100, 100, 100, cv2.BORDER_REPLICATE)
        img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REPLICATE)

        for f in range(4):
            rot_res = np.rot90(res, f)
            rot_img = np.rot90(img, f)
            bbox = detector_return(rot_res)
            if bbox != ():
                crop = write_image(bbox, rot_img)
                find, file = classify(crop)
                if find:
                    return find, file
    else:
        print 'image capture failed'
        return False, None

    return False, None


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    img_array = open(path, 'rb').read()
    print predict(img_array)
