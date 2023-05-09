from base64 import b64encode

import imageio
from PIL import Image
from keras.models import load_model
import keras.utils as image
import cv2
import numpy as np
import os

from Apply_Preset import get_edited_image

# imagepath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/test1.jpg'
modelpath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/EnhancementModel.h5'
#modelpath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/EnhancementModel



def pre_process_img(imagepath):

        up_img = Image.open(imagepath)
        up_img = up_img.resize((512, 512), Image.ANTIALIAS)
        up_array = image.img_to_array(up_img)
        up_array = np.expand_dims(up_array, axis=0)
        up_array /= 255.

        return up_array

# def enhance_img(imagepath):
#     print('hey')
#     processedImage = pre_process_img(imagepath)
#     og_img = Image.open(imagepath)
#     orig_image = np.asarray(og_img)
#     enh_model = load_model(modelpath,compile=False)
#     gen_image = enh_model.predict(processedImage)
#     gen_image = gen_image.reshape(gen_image.shape[1:])
#
#     final_image = cv2.resize(gen_image,
#                              (int(orig_image.shape[1]/2), int(orig_image.shape[0]/2)),
#                              interpolation=cv2.INTER_AREA)
#     final_image = cv2.normalize(final_image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
#
#     final_file = "enh_"
#
#     temp_path = os.path.join('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/FYP/Fyp_BE', 'imageoutput.jpg')
#
#     imageio.imwrite(temp_path, final_image)
#
#     return temp_path,final_image


def img_enh(imagepath):
    
    print('hey2')
    processedImage = pre_process_img(imagepath)
    og_img = Image.open(imagepath)
    orig_image = np.asarray(og_img)
    enh_model = load_model(modelpath,compile=False)

    #enhancing the image
    gen_image = enh_model.predict(processedImage)
    gen_image = gen_image.reshape(gen_image.shape[1:])

    final_image = cv2.resize(gen_image,
                             (int(orig_image.shape[1]/2), int(orig_image.shape[0]/2)),
                             interpolation=cv2.INTER_AREA)
    final_image = cv2.normalize(final_image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

    #rotated_image = final_image.rotate('90', expand=True)
    # final_file = "enh_"
    #
    # temp_path = os.path.join('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE', 'editedimage2.jpg')
    # #imagepath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/test1.jpg'
    #
    # imageio.imwrite(temp_path, rotated_image)
    #
    # return temp_path,rotated_image
    # final_file = "enh_"

    print("before print")
    print(final_image)
    print("print works")
    temp_path = os.path.join('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/FYP/Fyp_BE', 'editedimage1.jpg')
    imageio.imwrite(temp_path, final_image)
    
    # imagepath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/test1.jpg'

    img = cv2.imread(temp_path)

    # image = b64encode(img).decode("utf-8")



    return img

def img_preset(imagepath):
    print()
    print('hey3')
    processedImage = pre_process_img(imagepath)
    og_img = Image.open(imagepath)
    orig_image = np.asarray(og_img)

    result = get_edited_image(imagepath)

    # enh_model = load_model(modelpath_2,compile=False)

    # #enhancing the image
    # gen_image = enh_model.predict(processedImage)
    # gen_image = gen_image.reshape(gen_image.shape[1:])

    # final_image = cv2.resize(gen_image,
    #                          (int(orig_image.shape[1]/2), int(orig_image.shape[0]/2)),
    #                          interpolation=cv2.INTER_AREA)
    # final_image = cv2.normalize(final_image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

    #rotated_image = final_image.rotate('90', expand=True)
    # final_file = "enh_"
    #
    # temp_path = os.path.join('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE', 'editedimage2.jpg')
    # #imagepath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/test1.jpg'
    #
    # imageio.imwrite(temp_path, rotated_image)
    #
    # return temp_path,rotated_image
    # final_file = "enh_"

    print("before print")
    print(result)
    print("print works")

    temp_path = os.path.join('D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/FYP/Fyp_BE', 'presetImage1.jpg')
    imageio.imwrite(temp_path, result)
    
    # imagepath = 'D:/04.Uni_Related/L6/FYP/Photobuddy_Working_Version/Fyp_BE/test1.jpg'

    img = cv2.imread(temp_path)

    # image = b64encode(img).decode("utf-8")



    return img

#img_enh(imagepath)


