import os
import time
import uuid
import cv2
IMAGES_PATH = 'D:/Documents/DeepLearning/Image/Emotion/capture/surprise'
number_images = 200
cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    frame = frame[50:500, 50:500, :]
    cv2.imshow('frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
