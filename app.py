from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import urllib.request
import os
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ml as m


classify = []

#path = glob.glob("C:/Users/march/Desktop/intern/ml/static/uploads/*.jpg")

app = Flask(__name__, template_folder='template')

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

image_array = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('main.html')


@app.route('/ml')
def ml():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def upload_image():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        delete = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(delete)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')

        name = file.filename

        path = glob.glob(
            "C:/Users/march/Desktop/intern/ml/static/uploads/*.jpg")

        #i = Image.open(delete)

        img = cv2.imread(delete)

        '''
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''

        #image_sequence = img.getdata()

        #image_array = np.array(image_sequence)

        # print(image_array)

        # print(image_array.shape)

        result = m.classification(img)

        print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')

        print(result)

        global classify

        classify = result

        return render_template('index2.html', filename=filename, answer=classify)

        '''
        for delete in path:

            if str(name) == str(delete[(len(delete))-(len(str(name))):len(delete)]):

                print(str(name) + 'nnn')

                print(str(delete[(len(delete))-(len(str(name))):len(delete)]))

                print("Classify!")

                i = Image.open(delete)

                image_sequence = i.getdata()

                global image_array

                image_array = np.array(image_sequence)

                print(image_array)

                result = m.classification(image_array)

                print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')

                print(result)

                global classify

                classify = result

                break

            else:

                classify = name

                print(str(name))

                print(str(delete[(len(delete))-(len(str(name))):len(delete)]))

        return render_template('index2.html', filename=filename, answer=classify)
    '''

    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


'''

    if request.method == "POST":
        picture = request.form['picture']
        result = m.classification(picture)
        global classify
        classify = result
    return(render_template('index1.html', show=classify))'''


'''
@app.route("/sub", methods=['POST'])
def submit():
    # HTML >> .py
    if request.method == "POST":
        name = request.form["username"]

    # .py >> HTML
    return render_template("sub.html", n=name)
'''

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=80)
