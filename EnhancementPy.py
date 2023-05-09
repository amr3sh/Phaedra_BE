from mtcnn import MTCNN
import cv2
from deepface import DeepFace
from keras.models import load_model
from keras.preprocessing import image
from numpy import load
from numpy import expand_dims
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import tensorflow
# import keras.backend as tb
# import keras.backend.tensorflow_backend as tb
from keras import backend as K
from keras import losses
# import tensorflow as tf
# tb._SYMBOLIC_SCOPE.value = True

# nima_model=load_model('models/NIMA.h5')

class ImageUtils:

    def __init__(self, up_image):
        self.uploadedImage = up_image

    def validate_image(self):
        validate_image = cv2.imread(self.uploadedImage)
        face_detector = MTCNN()
        faces_n = face_detector.detect_faces(validate_image)
        print(len(faces_n))
        if len(faces_n) == 1:
            return True
        else:
            return False

    def pre_process_img(self):

        up_img = Image.open(self.uploadedImage)
        up_img = up_img.resize((512, 512), Image.ANTIALIAS)
        up_array = image.img_to_array(up_img)
        up_array = np.expand_dims(up_array, axis=0)
        up_array /= 255.

        return up_array


class EmotionDetector:

    def __init__(self, up_image):
        self.uploadedImage = up_image

    def getEmotion(self):
        demography = DeepFace.analyze(self.uploadedImage)
        final_emotion = demography['dominant_emotion']
        print(final_emotion)
        if final_emotion == "neutral":

            sad_perc = demography['emotion']['sad']
            happy_perc = demography['emotion']['happy']
            print(demography['emotion'])
            if sad_perc > happy_perc:
                final_emotion = "sad"
            else:
                final_emotion = "happy"
        elif final_emotion == "sad" or final_emotion == "happy":
            final_emotion = final_emotion
        else:
            final_emotion = "invalid"

        return final_emotion


def mean_score(scores):
    si = K.arange(1, 11, 1)
    sc = K.cast(scores, 'float32')
    si = K.cast(si, 'float32')
    mean = K.sum(sc * si)
    return mean


def std_score(scores):
    si = K.arange(1, 11, 1)
    mean = mean_score(scores)
    si = K.cast(si, 'float32')
    mean = K.cast(mean, 'float32')
    std = K.sqrt(K.sum(((si - mean) ** 2) * scores))
    return std

#
# def NIMA_Loss(y_true, y_pred):
#     gamma = 0.0001
#     # Pre-processing y-pred before sending to the NIMA model
#     num_ex = K.shape(y_pred)[0]
#     y_img = tf.image.resize(y_pred, (224, 224))
#     # Getting Predicted score from NIMA model
#     y_p = nima_model(K.reshape(y_img, (num_ex, 224, 224, 3)))
#     scores = y_p
#     # Getting Final predicted score
#     finalScore = mean_score(scores) + std_score(scores)
#     # Getting Final Loss
#     finalLoss = losses.mean_absolute_error(y_true, y_pred) + gamma * (10 - finalScore)
#
#     return finalLoss


class ImageEnhancer:

    def __init__(self, up_image, image_utils):

        self.processedImage = image_utils.pre_process_img()
        self.happypath = 'models/EnhModelHappyFinal.h5'
        self.sadpath = 'models/EnhModelsadFinal.h5'

        og_img = Image.open(up_image)
        self.orig_image = np.asarray(og_img)

    def enhance_image(self, sent_parameter, savepath, o_filename):

        if sent_parameter == "happy":
            happyModel = load_model(self.happypath, compile=False)
            gen_image = happyModel.predict(self.processedImage)
            gen_image = gen_image.reshape(gen_image.shape[1:])
        elif sent_parameter == "sad":
            sadModel = load_model(self.sadpath, compile=False)
            gen_image = sadModel.predict(self.processedImage)
            gen_image = gen_image.reshape(gen_image.shape[1:])

        final_image = cv2.resize(gen_image,
                                 (int(self.orig_image.shape[1] / 2), int(self.orig_image.shape[0] / 2)),
                                 interpolation=cv2.INTER_AREA)
        final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        final_name = "enh_" + o_filename

        temp_path = os.path.join(savepath, final_name)

        imageio.imwrite(temp_path, final_image)

        return temp_path



class ImageComparison:

    def __init__(self, original_image, enhanced_image):

        self.originalImage = cv2.imread(original_image)
        self.enhancedImage = cv2.imread(enhanced_image)
        self.hsv_og = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2HSV)
        self.hsv_enh = cv2.cvtColor(self.enhancedImage, cv2.COLOR_BGR2HSV)

    def get_brightness(self):

        _, _, v = cv2.split(self.hsv_og)
        _, _, v1 = cv2.split(self.hsv_enh)

        og_brightness = int(np.average(v.flatten()))
        enh_brightness = int(np.average(v1.flatten()))

        brightness_change = enh_brightness - og_brightness

        if brightness_change > 0:
            bright_str = "+ " + str(brightness_change)
        else:
            bright_str = str(brightness_change)

        return bright_str

    def get_hue(self):

        h, _, _ = cv2.split(self.hsv_og)

        h1, _, _ = cv2.split(self.hsv_enh)

        og_hue = int(np.average(h.flatten()))
        enh_hue = int(np.average(h1.flatten()))

        hue_change = enh_hue - og_hue

        if hue_change > 0:
            hue_str = "+ " + str(hue_change)
        else:
            hue_str = str(hue_change)

        return hue_str

    def get_saturation(self):

        _, s, _ = cv2.split(self.hsv_og)

        _, s1, _ = cv2.split(self.hsv_enh)

        og_saturation = int(np.average(s.flatten()))
        enh_saturation = int(np.average(s1.flatten()))

        saturation_change = enh_saturation - og_saturation

        if saturation_change > 0:
            saturation_str = "+ " + str(saturation_change)
        else:
            saturation_str = str(saturation_change)

        return saturation_str

    def get_contrast(self):

        lab_og = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2LAB)
        lab_enh = cv2.cvtColor(self.enhancedImage, cv2.COLOR_BGR2LAB)

        L, _, _ = cv2.split(lab_og)

        L1, _, _ = cv2.split(lab_enh)

        kernel = np.ones((5, 5), np.uint8)
        min = cv2.erode(L, kernel, iterations=1)
        max = cv2.dilate(L, kernel, iterations=1)

        min = min.astype(np.float64)
        max = max.astype(np.float64)

        contrast = (max - min) / (max + min)

        average_contrast_og = 100 * np.mean(contrast)

        kernel = np.ones((5, 5), np.uint8)
        min = cv2.erode(L1, kernel, iterations=1)
        max = cv2.dilate(L1, kernel, iterations=1)

        min = min.astype(np.float64)
        max = max.astype(np.float64)

        contrast = (max - min) / (max + min)

        average_contrast_enh = 100 * np.mean(contrast)

        contrast_change = average_contrast_enh - average_contrast_og

        if contrast_change > 0:
            contrast_str = "+ " + str(int(contrast_change))
        else:
            contrast_str = str(int(contrast_change))

        return contrast_str

    def get_histogram(self, chartpath):
        vals = self.enhancedImage.mean(axis=2).flatten()
        b, bins, patches = plt.hist(vals, 255)
        plt.xlim([0, 255])
        temp_path = os.path.join(chartpath, 'histogram.png')
        plt.savefig(temp_path)

        return temp_path
