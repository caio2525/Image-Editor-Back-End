from flask import Flask, request, make_response, send_file
from flask_cors import CORS
import numpy as np
import cv2
import base64
import urllib
from io import BytesIO
from utils import fromByteToImg, fromImgToBase64, fromBase64ToImg
from imageProcessing import addGaussianNoise, add_salt_pepper_noise, boxFiltro, gaussianFiltro, medianFiltro, unsharpMascara

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello World, Caio!</p>"

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files['image']
        #read image file string data
        filestr = request.files['image'].read()
        img = fromByteToImg(filestr)

        img_base64 = fromImgToBase64(img)
        response = make_response(img_base64)
        response.headers.set('Content-Type', 'multipart/form-data')

        return response

    except:
        return 'bad request!', 400

@app.route("/gaussianNoise", methods=["POST"])
def gaussianNoise():
    json_data = request.json
    #print(json_data)
    #print(type(json_data))
    print(json_data['Sigma'])
    #print(json_data['image'])
    img = fromBase64ToImg(json_data['image'])
    noisyImage = addGaussianNoise(img, int(json_data['Sigma']))


    img_base64 = fromImgToBase64(noisyImage)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return(response)

@app.route("/saltPepperNoise", methods=["POST"])
def saltPepperNoise():
    json_data = request.json
    #print(json_data)
    #print(type(json_data))
    print(json_data['Amount'])
    #print(json_data['image'])
    img = fromBase64ToImg(json_data['image'])
    noisyImage = add_salt_pepper_noise(img, amount=float(json_data['Amount']))

    img_base64 = fromImgToBase64(noisyImage)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return(response)

@app.route("/boxFilter", methods=["POST"])
def boxFilter():
    json_data = request.json
    print('KernelSize', json_data['KernelSize'])

    img = fromBase64ToImg(json_data['image'])
    blurredImg = boxFiltro(img, kernelSize=int(json_data['KernelSize']))

    img_base64 = fromImgToBase64(blurredImg)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return(response)

@app.route("/gaussianFilter", methods=["POST"])
def gaussianFilter():
    json_data = request.json
    print('KernelSize', json_data['KernelSize'])
    print('Sigma', json_data['Sigma'])

    img = fromBase64ToImg(json_data['image'])
    blurredImg = gaussianFiltro(img, kernelSize=int(json_data['KernelSize']), sigma=int(json_data['Sigma']))

    img_base64 = fromImgToBase64(blurredImg)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return(response)

@app.route("/medianFilter", methods=["POST"])
def medianFilter():
    json_data = request.json
    print('KernelSize', json_data['KernelSize'])

    img = fromBase64ToImg(json_data['image'])
    filteredImage = medianFiltro(img, kernelSize=int(json_data['KernelSize']))

    img_base64 = fromImgToBase64(filteredImage)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return(response)

@app.route("/unsharpMask", methods=["POST"])
def unsharpMask():
    json_data = request.json
    print('KernelSize', json_data['KernelSize'])

    img = fromBase64ToImg(json_data['image'])
    sharperImage = unsharpMascara(img, kernelSize=int(json_data['KernelSize']))

    img_base64 = fromImgToBase64(sharperImage)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return(response)

@app.route("/button", methods=["POST"])
def button():

    img = fromBase64ToImg(request.data)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    img = cv2.putText(img, 'OpenCV', org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

    img_base64 = fromImgToBase64(img)
    response = make_response(img_base64)
    response.headers.set('Content-Type', 'multipart/form-data')

    return response


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
