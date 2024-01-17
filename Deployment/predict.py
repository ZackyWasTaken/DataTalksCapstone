import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from werkzeug.datastructures import FileStorage
import tensorflow as tf
from tensorflow import keras
from io import BytesIO
from PIL import Image

model = keras.models.load_model('xception_final_139_0.820_0.936.h5')

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.applications.xception import preprocess_input

app = Flask('cat')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    if isinstance(file, FileStorage):

        img_bytes = file.read()
        image = Image.open(BytesIO(img_bytes))
        img_bytes_io = BytesIO()
        image.save(img_bytes_io, format='JPEG')
        img_bytes_io.seek(0)

        img = load_img(img_bytes_io, target_size=(299, 299))

        x = np.array(img)
        X = np.array([x])
        X = preprocess_input(X)
        pred = model.predict(X)
        pred_list = pred.tolist()
        classes = [
        'Abyssinian' ,'American Bobtail' ,'American Curl' ,'American Shorthair' ,'Bengal' ,'Birman' ,'Bombay' ,
        'British Shorthair' ,'Egyptian Mau' ,'Exotic Shorthair' ,'Maine Coon' ,'Manx' ,'Norwegian Forest' ,'Persian' ,
        'Ragdoll' ,'Russian Blue' ,'Scottish Fold' ,'Siamese' ,'Sphynx' ,'Turkish Angora']
        
        result = dict(zip(classes, pred_list[0]))

        return jsonify(result)


if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=9696)