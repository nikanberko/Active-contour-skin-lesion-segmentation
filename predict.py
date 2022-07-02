
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, send_file
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from werkzeug.utils import secure_filename, redirect

app = Flask(__name__)





def segmentTumor(tumorPath, contourAlpha, contourBeta, contourGamma ):
    print("Loading object detector")
    model = load_model("static/mlmodels/detector.h5")

    imagePath = tumorPath

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    (rx, ry, cx, cy) = preds

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    rx = int(rx * w)
    ry = int(ry * h)
    cx = int(cx * w * 1.3)
    cy = int(cy * h * 1.3)

    img = image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    s = np.linspace(0, 360, 360)

    r = rx + cx * np.cos(np.radians(s))

    c = ry + cy * np.sin(np.radians(s))

    init = np.array([c, r]).T

    snake = active_contour(gaussian(thresh, 3, preserve_range=False), init, alpha=contourAlpha, beta=contourBeta, gamma=contourGamma)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis('off')
    plt.savefig("static/uploads/segmentedTumor.jpeg", bbox_inches='tight', pad_inches=0)






basePathToImage= "static/uploads"
dir="static/uploads"
for file in os.listdir(dir):
    os.remove(os.path.join(dir,file))
    

@app.route("/")
def uploader():
        path = 'static/uploads/'
        uploads = sorted(os.listdir(path), key=lambda x: os.path.getctime(path+x))
        print(uploads)
        uploads = ['uploads/' + file for file in uploads]
        uploads.reverse()
        return render_template("index.html",uploads=uploads)

app.config['UPLOAD_PATH'] = 'static/uploads'
@app.route("/upload",methods=['GET','POST'])
def upload_file():
        if request.method == 'POST':
                dir="static/uploads"
                for file in os.listdir(dir):
                    os.remove(os.path.join(dir,file))
                f = request.files['file']
                print(f.filename)
                filename = f.filename
                f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
                uploadedImagePath= basePathToImage+"/"+f.filename;
                segmentTumor(uploadedImagePath, 0.009, 0.5, 0.0001)
                return redirect("/")

@app.route('/download')
def download_file():
	path = "static/uploads/segmentedTumor.jpeg"
	return send_file(path, as_attachment=True)

if __name__=="__main__":
    app.run()