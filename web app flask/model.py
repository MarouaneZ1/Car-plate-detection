import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

model = tf.keras.models.load_model('./static/models/object_detection2.h5')

def object_detection(path,filename):
    imag = load_img(path)
    image = img_to_array(imag)
    image1 = load_img(path, target_size=(224,224))
    image_arr = img_to_array(image1)
    image_arr_norm = image_arr/255.0
    h,w,d = image.shape
    test_arr = image_arr_norm.reshape(1,224,224,3)
    pred = model.predict(test_arr)
    pred = pred[0]
    xmin, xmax, ymin, ymax = pred[0]*w, pred[1]*w, pred[2]*h, pred[3]*h
    pred_denorm = np.array((xmin, xmax, ymin, ymax)).astype(np.int32)
    xmin, xmax, ymin, ymax = pred_denorm
    pixels = np.array(imag)
    cv2.rectangle(pixels, (pred_denorm[0], pred_denorm[2]), (pred_denorm[1], pred_denorm[3]), (0,255,0), 3)
    #plt.imshow(pixels)
    """imges = cv2.imread(path)
    gg = cv2.cvtColor(imges,cv2.COLOR_RGB2BGR)
    pixels = np.array(gg)"""
    #image_bgr = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./static/predict/{}'.format(filename),pixels)
    return pixels, pred_denorm

def extract_plac_text(path,filename):
    img, prediction = object_detection(path,filename)
    xmin, xmax, ymin, ymax = prediction
    plac = img[ymin:ymax, xmin:xmax]
    #roi_bgr = cv2.cvtColor(plac,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename),plac)
    text = pt.image_to_string(plac)
    return plac, text