from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model_mlnn = load_model('model_MLNN.h5')
model_cnn = load_model('model_CNN.h5')
model_cnndo = load_model('model_CNNDO.h5')

class_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8:'8', 9: '9'}

def predict_label(img_path, model, nama_model):
    loaded_img = load_img(img_path, target_size=(400, 400, 3))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    pred = model.predict(img_array)
    prediksi = np.argmax(pred)
    print(nama_model, ' memprediksi: ', prediksi)
    predicted_bit = np.round(prediksi).astype('int')
    return class_dict[predicted_bit]

@app.route('/', methods=['GET', 'POST'])
def index():
    vals = []
    angka_sebenarnya = ''
    model_dl = np.nan

    if request.method == 'POST':
        vals = [x for x in request.form.values()]
        angka_sebenarnya = vals[0]
        model_dl = vals[1]

        if (model_dl == 'MLNN'):
            nama_model = 'MLNN'
            model = model_mlnn
        if (model_dl == 'CNN'):
            nama_model = 'CNN'
            model = model_cnn
        if (model_dl == 'CNNDO'   ):
            nama_model = 'CNN with DO'
            model = model_cnndo
        print('modelnya: ', nama_model, 'summary: ',  model.summary())

        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path, model, nama_model)
            print(img_path)
            akurat = 'Akurat, sama dengan angka sebenarnya ' + angka_sebenarnya
            if str(angka_sebenarnya) != prediction:
                akurat = 'Meleset, angka sebenarnya ' + angka_sebenarnya
            print(akurat)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction, model=nama_model, akurat=akurat)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)