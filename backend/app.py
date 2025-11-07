import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# -----------------------
# Configuration
# -----------------------
BASE_DIR = '/home/ashutosh/Desktop/Smart Image Caption Generator/backend'   # change if needed
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# -----------------------
# Utility helpers
# -----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def idx_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer, None)

# -----------------------
# Load model & objects once
# -----------------------
print("Loading model and artefacts...")
model = load_model(os.path.join(BASE_DIR, 'final_model.h5'))
with open(os.path.join(BASE_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)
with open(os.path.join(BASE_DIR, 'max_length.pkl'), 'rb') as f:
    max_length = pickle.load(f)

# VGG16 feature extractor (same as training)
vgg = VGG16()
vgg = VGG16()
vgg = VGG16()
# restructure to get penultimate layer
from tensorflow.keras.models import Model # type: ignore
vgg = VGG16()
vgg_fe = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

print("Loaded model, tokenizer and VGG feature-extractor.")

# -----------------------
# Caption generation
# -----------------------
def generate_caption_from_feature(feature, model, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def extract_features_from_image(image: Image.Image):
    image = image.resize((224, 224))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = vgg_fe.predict(arr, verbose=0)
    return feat.reshape((1, feat.shape[-1]))  # shape (1, 4096)

# -----------------------
# Routes
# -----------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    # check file
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = secure_filename(file.filename) # type: ignore
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # load image and extract features
    image = Image.open(save_path).convert('RGB')
    feature = extract_features_from_image(image)  # shape (1, 4096)

    # generate caption
    caption = generate_caption_from_feature(feature, model, tokenizer, max_length)

    # clean caption (remove startseq/endseq)
    caption_clean = caption.replace('startseq', '').replace('endseq', '').strip()

    return render_template('result.html', filename=filename, caption=caption_clean)

# -----------------------
# Run
# -----------------------
if __name__ == '__main__':
    # set host=0.0.0.0 if you want external access on LAN
    app.run(debug=True, port=5000)
