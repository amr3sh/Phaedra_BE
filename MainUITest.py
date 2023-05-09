from flask import Flask, render_template, request, send_from_directory
import os
from EnhancementPy import ImageComparison, ImageUtils, ImageEnhancer, EmotionDetector
from werkzeug.utils import secure_filename
import glob

UPLOAD_FOLDER = os.path.join('static', 'temp_upload')
OUTPUT_FOLDER = os.path.join('static', 'enhance_output')
HIST_FOLDER = os.path.join('static', 'histogram_plot')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['HIST_FOLDER'] = HIST_FOLDER

m_filepath = " "
m_filename = " "
img_utils = None


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route('/')
def index():
    return render_template("Main.html")


@app.route('/EnhanceImage', methods=['POST'])
def enhance_image():
    img_enhance = ImageEnhancer(m_filepath, img_utils)
    emotion_detect = EmotionDetector(m_filepath)

    scanned_emotion = emotion_detect.getEmotion()

    print(scanned_emotion)

    if scanned_emotion == "invalid":
        return render_template("Main.html", invalid=True, errormessage="Image cannot be enhanced for detected emotion")
    else:
        enhanced_image = img_enhance.enhance_image(scanned_emotion, app.config['OUTPUT_FOLDER'], m_filename)
        image_compare = ImageComparison(m_filepath, enhanced_image)
        brightness_v = image_compare.get_brightness()
        saturation_v = image_compare.get_saturation()
        hue_v = image_compare.get_hue()
        contrast_v = image_compare.get_contrast()
        histogram_path=image_compare.get_histogram(app.config['HIST_FOLDER'])
        return render_template("EnhanceUI.html", filepath=enhanced_image, ogfilepath=m_filepath,
                               brightness=brightness_v, hue=hue_v, saturation=saturation_v, contrast=contrast_v, histpath=histogram_path)

    # return render_template("EnhanceUI.html")


@app.route('/ValidateImage', methods=['POST'])
def ValidateImage():
    global img_utils
    global m_filepath, m_filename
    if request.method == 'POST':
        file = request.files['file']
        print(file.name)
        filename = secure_filename(file.filename)
        m_filename = filename

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        m_filepath = filepath
        file.save(filepath)
        img_utils = ImageUtils(filepath)

        if img_utils.validate_image():
            print(filepath)
            return render_template("Main.html", filepath=filepath)
        else:
            return render_template("Main.html", invalid=True,
                                   errormessage="Image validation has failed due to more than one face being detected")


if __name__ == "__main__":
    app.run(threaded=False)
