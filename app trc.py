import flask
import tensorflow as tf
import numpy as np
from flask import Flask

app = Flask(__name__)

def predict_leaf_disease(image):
    with tf.Session() as sess:
        model = tf.keras.models.load_model('batsgirls-leaves.h5')
        prediction = model.predict(image)
        disease = tf.keras.backend.argmax(tf.keras.losses.sparse_categorical_crossentropy(y_true=np.zeros(prediction.shape), y_pred=prediction))
        return disease

def predict_fruit_maturity(image):
    with tf.Session() as sess:
        model = tf.keras.models.load_model('Tomato_ripeness1.h5')
        prediction = model.predict(image)
        maturity = prediction.argmax()
        return maturity

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image'].read()
    disease = predict_leaf_disease(image)
    maturity = predict_fruit_maturity(image)
    result = {
        'disease': disease,
        'maturity': maturity
    }
    return jsonify(result)