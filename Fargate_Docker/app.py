from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten
from tensorflow.keras import Sequential

app = Flask(__name__)

# Load the trained models
pneumonia_model = None
bone_age_model = None
brain_tumor_model = None

global std_bone_age
global mean_bone_age

std_bone_age = 41.18202139939618
mean_bone_age = 127.32041884816753

def load_penumonia_model():
    global pneumonia_model
    pneumonia_model_path = 'pneumonia_detection_ai_version_3.h5'
    pneumonia_model = load_model(pneumonia_model_path)

def load_bone_age_model():
    global bone_age_model
    bone_age_model_path = 'bone_age.h5'
    model_1 = tensorflow.keras.applications.xception.Xception(input_shape = (256, 256, 3),
                                            include_top = False,
                                            weights = 'imagenet')
    model_1.trainable = True
    bone_age_model = Sequential()
    bone_age_model.add(model_1)
    bone_age_model.add(GlobalMaxPooling2D())
    bone_age_model.add(Flatten())
    bone_age_model.add(Dense(10, activation = 'relu'))
    bone_age_model.add(Dense(1, activation = 'linear'))

    #compile model
    bone_age_model.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months])
    bone_age_model.load_weights(bone_age_model_path)

def load_brain_tumor_model():
    global brain_tumor_model
    brain_tumor_model_path = 'Brain_Tumors.h5'
    brain_tumor_model = load_model(brain_tumor_model_path)

def preprocess_image_pneumonia(image):
    img_arr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_arr = cv2.resize(img_arr, (200, 200))
    normalized_img = resized_arr / 255.0
    reshaped_img = np.reshape(normalized_img, (1, 200, 200, 1))
    return reshaped_img

def preprocess_image_bone_age(image):
    # Preprocess the image for bone age model
    resized_img = cv2.resize(image, (256, 256))  # Resize as per bone age model requirement
    img_array = np.array(resized_img)
    preprocessed_img = preprocess_input(img_array)  # Use Xception preprocessing
    return np.expand_dims(preprocessed_img, axis=0)

def predict_pneumonia(image):
    if pneumonia_model is None:
        load_penumonia_model()
    preprocessed_image = preprocess_image_pneumonia(image)
    prediction = pneumonia_model.predict(preprocessed_image)
    return prediction

def preprocess_image_brain_tumor(image):
    resized_img = cv2.resize(image, (256, 256))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_array = tensorflow.keras.preprocessing.image.img_to_array(resized_img)
    return tensorflow.expand_dims(img_array, 0)

def mae_in_months(x_p, y_p):
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age))

def predict_brain_tumor(image):
    if brain_tumor_model is None:
        load_brain_tumor_model()
    preprocessed_image = preprocess_image_brain_tumor(image)
    prediction = brain_tumor_model.predict(preprocessed_image)
    return prediction

def predict_bone_age(image):
    if bone_age_model is None:
        load_bone_age_model()
    preprocessed_image = preprocess_image_bone_age(image)
    prediction = mean_bone_age + std_bone_age*(bone_age_model.predict(preprocessed_image))
    return prediction

@app.route('/predict_pneumonia', methods=['POST'])
def predict_pneumonia_api():
    try:
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(
            image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        prediction = predict_pneumonia(image)
        result = {'prediction': prediction[0][0]}
        if prediction[0][0] > 0.7:
            result = {'pneumonia_prediction': 'Normal'}
        else:
            result = {'pneumonia_prediction': 'Pneumonia'}
        return jsonify({"result": str(result)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_bone_age', methods=['POST'])
def predict_bone_age_api():
    try:
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(
            image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        prediction = predict_bone_age(image)
        result = {'bone_age_prediction': prediction[0][0]/12.0}
        return jsonify({"result": str(result)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_brain_tumor', methods=['POST'])
def predict_brain_tumor_api():
    try:
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(
            image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        prediction = predict_brain_tumor(image)
        score = tensorflow.nn.softmax(prediction[0])
        class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class = class_labels[tensorflow.argmax(score)]
        return jsonify({"result": str(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
