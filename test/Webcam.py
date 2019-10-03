import numpy as np
import cv2
from PIL import Image, ImageOps

import util.ImageLoader as ImageLoader
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # print(frame.shape)

    image = Image.fromarray(frame)
    image = ImageLoader.center_image_with_padding(image, 1920, 1080)
    image = np.array(image)

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()