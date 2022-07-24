from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from sympy import per
import tensorflow as tf
facetracker = load_model(
    './facetrackerFinal2.h5', compile=False)
classifier = load_model(
    './my_model_final_final_final_final_final_final.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip camera vertically
    frame = frame[50:500, 50:500, :]
    data = np.zeros((200, 200, 3), dtype="uint8")
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
        roi_gray = gray[y-10:h+5, x:w]
        percent = []
        try:
            roi = cv2.resize(roi_gray, (48, 48),
                             interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                pos = prediction.argmax()
                label = emotion_labels[pos]
                percent_position = [0, 0]
                for i in range(len(prediction)):
                    percent.append(str(round((prediction[i]*100), 2)))
                    percent_position = [0, (i+2)*20]
                    cv2.putText(data, emotion_labels[i] + ':' + percent[i],
                                percent_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        except Exception as e:
            print(e)
        cv2.rectangle(frame, tuple(sample_coords[:2]), tuple(
            sample_coords[2:]), (255, 0, 0), 2)
    cv2.imshow('face', frame)
    cv2.imshow('data', data)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
