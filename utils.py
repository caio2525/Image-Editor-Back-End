import numpy as np
import cv2
import base64

def fromByteToImg(buffer):
    #convert string data to numpy array
    npimg = np.frombuffer(buffer, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, 0)

    return(img)

def fromImgToBase64(img):
    retval, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer)
    return (img_base64)

def fromBase64ToImg(base64Data):
    decoded_data = base64.b64decode(base64Data)
    npimg = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(npimg, 0)
    return(img)
