import subprocess
import os
import json
from PIL import Image

from flask import Flask, render_template,request,jsonify
from flask_cors import CORS
import joblib
app = Flask(__name__)
cors = CORS(app)
import pickle
from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np
import cv2
import base64

from flask import Flask, render_template, request, send_from_directory
import os
from flask_cors import CORS
#from EnhTest import pre_process_img
#from werkzeug.utils import secure_filename
#import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model
from EnhTest import img_enh
from EnhTest import img_preset
from new_classifier_code import keras_train_classify
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import send_file

import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS


UPLOAD_FOLDER = os.path.join('static', 'temp_upload')
OUTPUT_FOLDER = os.path.join('static', 'enhance_output')
HIST_FOLDER = os.path.join('static', 'histogram_plot')

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['HIST_FOLDER'] = HIST_FOLDER

m_filepath = " "
m_filename = " "
img_utils = None
# imagepath = 'D:/FYP/Phaedra_BE/65d.jpg'
finalImgFilePath = 'D:/FYP/FYP/Fyp_BE'

def meta_data_extract(imagepath):
    infoDict = {}  # Creating the dict to get the metadata tags
    exifToolPath = imagepath  # for Windows user have to specify the Exif tool exe path for metadata extraction.
    # For mac and linux user it is just
    """exifToolPath = exiftool"""


    imgPath = imagepath

    ''' use Exif tool to get the metadata '''
    process = subprocess.Popen([exifToolPath, imgPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
    ''' get the tags in dict '''
    for tag in process.stdout:
        line = tag.strip().split(':')
        infoDict[line[0].strip()] = line[-1].strip()

    for k, v in infoDict.items():
        print(k, ':', v)

    return infoDict.items();

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-store"
    return response



@app.route('/EnhanceImage', methods=['POST'])
def enhance_image():
    imgUrl = request.files["photo"]
    print(imgUrl)
    # creating a image object (main image)
    im1 = Image.open(imgUrl)

    # save a image using extension
    im1 = im1.save('D:/FYP/Phaedra_BE/capturedImage.jpg', 'JPEG')
    

    imagepath = "D:/FYP/Phaedra_BE/capturedImage.jpg"
    #imagepath = ('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/FYP/capturedImage.jpg' , mimetype = 'image/jpg')
    #calling the enhancement method
    image = img_enh(imagepath)
    
    # print(image)
    #works till this point

    img = Image.fromarray(image.astype("uint8"))
    
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    # im = Image.open(image)
    # data = io.BytesIO()
    # im.save(data, "JPEG")
    # encoded_img_data = base64.b64encode(data.getvalue())
    # return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))
    return jsonify({'status': str(img_base64)})


    #result_fe = jsonify({"image": image})

    #return send_file("image.jpg", mimetype='image/jpg')
    # img_enhance = ImageEnhancer(m_filepath, img_utils)
    # return render_template("EnhanceUI.html")


@app.route('/PresetImage', methods=['POST'])
def preset_image():
    imgUrl = request.files["photo"]
    print(imgUrl)
    # creating a image object (main image)
    im1 = Image.open(imgUrl)

    # save a image using extension
    im1 = im1.save('D:/FYP/Phaedra_BE/capturedImage.jpg', 'JPEG')
    

    imagepath = "D:/FYP/Phaedra_BE/capturedImage.jpg"
    #imagepath = ('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/FYP/capturedImage.jpg' , mimetype = 'image/jpg')
    #calling the enhancement method
    image = img_preset(imagepath)
    
    # print(image)
    #works till this point

    img = Image.fromarray(image.astype("uint8"))
    
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    # im = Image.open(image)
    # data = io.BytesIO()
    # im.save(data, "JPEG")
    # encoded_img_data = base64.b64encode(data.getvalue())
    # return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))
    return jsonify({'status': str(img_base64)})


# @app.route('/metaData', methods=['POST'])
# def enhance_image():
#     imagepath = request.json['path']
#     img_enhance = enhance_image(imagepath)
#     # img_enhance = ImageEnhancer(m_filepath, img_utils)
#     # return render_template("EnhanceUI.html")

@app.route('/qualityCheck', methods=['POST'])
def img_quality_check():
    # img_enhance = ImageEnhancer(m_filepath, img_utils)
    # return render_template("EnhanceUI.html")
    # Load model

    imgUrl = request.files["photo"]
    print(imgUrl)
    # creating a image object (main image)
    im1 = Image.open(imgUrl)

    # save a image using extension
    im1 = im1.save('D:/FYP/Phaedra_BE/capturedImage.jpg', 'JPEG')

    imagepath = "D:/FYP/Phaedra_BE/capturedImage.jpg"

    model = load_model('D:/FYP/Phaedra_BE/resnet50.h5')
    # Load Image
    data = []
    image_filename = imagepath
    image = cv2.imread(image_filename)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    data.append(image)
    imgs = np.asarray(data).astype('float32')
    print(imgs.shape)

    # Predict on Model
    predictionss = model.predict(imgs)
    predictions = keras_train_classify(imagepath)
    pred = predictions.tolist()
    final_pred = pred[0]
    print(predictions)
    print(pred)

    print(final_pred)

    if (final_pred[0] > final_pred[1] and final_pred[0] > final_pred[2] and final_pred[0] > final_pred[3] and
            final_pred[0] > final_pred[4]):
        result = "Correct"
    elif (final_pred[1] > final_pred[0] and final_pred[1] > final_pred[2] and final_pred[1] > final_pred[3] and
          final_pred[1] > final_pred[4]):
        result = "Shutter Speed High"
    elif (final_pred[2] > final_pred[0] and final_pred[2] > final_pred[1] and final_pred[2] > final_pred[3] and
          final_pred[2] > final_pred[4]):
        result = "Shutter Speed Low"
    elif (final_pred[3] > final_pred[0] and final_pred[3] > final_pred[1] and final_pred[3] > final_pred[2] and
          final_pred[3] > final_pred[4]):
        result = "ISO Low"
    elif (final_pred[4] > final_pred[0] and final_pred[4] > final_pred[1] and final_pred[4] > final_pred[2] and
          final_pred[4] > final_pred[3]):
        result = "ISO High"

    # if(final_pred[0] > final_pred[1] and final_pred[0] > final_pred[2] and final_pred[0] > final_pred[3] and final_pred[0] > final_pred[4]):
    #     result = "ISO Low"
    # elif(final_pred[1] > final_pred[0] and final_pred[1] > final_pred[2] and final_pred[1] > final_pred[3] and final_pred[1] > final_pred[4]):
    #     result = "Correct"
    # elif (final_pred[2] > final_pred[0] and final_pred[2] > final_pred[1] and final_pred[2] > final_pred[3] and final_pred[2] > final_pred[4]):
    #     result = "Shutter Speed Low"
    # elif (final_pred[3] > final_pred[0] and final_pred[3] > final_pred[1] and final_pred[3] > final_pred[2] and final_pred[3] > final_pred[4]):
    #     result = "ISO High"
    # elif (final_pred[4] > final_pred[0] and final_pred[4] > final_pred[1] and final_pred[4] > final_pred[2] and final_pred[4] > final_pred[3]):
    #     result = "Shutter Speed High"

    #meta_data = meta_data_extract(imagepath);
    print(predictions)
    print(result)
    keras_result = keras_train_classify(imagepath)
    print(keras_result)
    result_fe = jsonify({"result": result})

    return result_fe

if __name__ == "__main__":
    app.run(threaded=False)
