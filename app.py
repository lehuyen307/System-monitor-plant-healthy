import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request, render_template
from keras.preprocessing.image import img_to_array, load_img
from flask_cors import CORS

def preprocess_img(img):
    img = load_img(img)
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, (200, 200))
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_class(prediction):
    predicted_class = np.argmax(prediction, axis=-1)
    Disease = ['Bac la muon', 'Bac la som', 'Khoe manh', 'Moc la', 'Nhen ve hai dom',
                   'Vang xoan la', 'Kham la', 'Diem muc tieu', 'Dom la nau', 'Dom vi khuan']
    if predicted_class < 0 or predicted_class >= len(Disease):
        return "Bệnh không xác định"
    else:
        return Disease[predicted_class]
    
# import request
app = Flask(__name__)
CORS(app)
model1 = tf.keras.models.load_model('batsgirls-leaves.h5')
model2 = tf.keras.models.load_model('Tomato_ripeness1.h5')

def predict_leaf_disease(image):
    # Load the model

    # Predict using the model
    prediction = model1.predict(image)
    # Get the predicted disease class label
    predicted_disease = predict_class(prediction)
    return predicted_disease

def predict_fruit_maturity(image):

    # Preprocess image if needed based on the model's input requirements
    prediction = model2.predict(image)
    maturity = np.argmax(prediction)
    return maturity


@app.route('/', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image']
    image.save('uploaded_image.jpg')

    predicted_disease = predict_leaf_disease('uploaded_image.jpg')

    result = {
        'disease': predicted_disease,
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)


