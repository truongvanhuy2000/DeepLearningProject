from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from sympy import per
import tensorflow as tf
import time
import uuid
import os
facetracker = load_model(
    './facetrackerFinal2.h5', compile=False)
IMAGES_PATH = 'D:/Documents/DeepLearning/Image/Emotion/capture/angry'
number_images = 200
cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip camera vertically
    frame = frame[50:500, 50:500, :]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]
    if yhat[0] > 0.8:
        # Controls the main rectangle
        sample_coords = np.multiply(sample_coords, 450).astype(int)
        # Print emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = sample_coords
        roi_gray = gray[y-10:h+10, x-5:w+5]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, tuple(sample_coords[:2]), tuple(
            sample_coords[2:]), (255, 0, 0), 2)
    cv2.imshow('face', frame)
    cv2.imshow('data', roi_gray)
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, roi_gray)
    time.sleep(0.2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()