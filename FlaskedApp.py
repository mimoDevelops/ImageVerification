from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.activations import sigmoid

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('C:/Users/jaf23/MachAndPy/ImageClass/fine_tuned_model')

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Image Classification Service</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f7f7f7;
                    text-align: center;
                    padding-top: 50px;
                }
                h1 {
                    color: #333;
                }
                .upload-form {
                    background-color: #fff;
                    margin: auto;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    width: 300px;
                }
                .upload-btn {
                    margin: 10px 0;
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                    background-color: #4CAF50;
                    color: white;
                    cursor: pointer;
                }
                .upload-btn:hover {
                    background-color: #45a049;
                }
                .upload-input {
                    width: 100%;
                    padding: 10px;
                    margin: 5px 0;
                    display: block;
                    border-radius: 4px;
                    box-sizing: border-box;
                }
            </style>
        </head>
        <body>
            <h1>Image Classification Service</h1>
            <div class="upload-form">
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" class="upload-input">
                    <input type="submit" value="Upload" class="upload-btn">
                </form>
            </div>
        </body>
    </html>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((160, 160))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 160, 160, 3))

        # Make a prediction
        prediction = model.predict(img_array)
        probability = sigmoid(prediction).numpy()

        # Format the response
        response = {
            "prediction": str(probability[0][0]),
            "confidence": f"{probability[0][0] * 100:.2f}%",
            "class": "Cat" if probability[0][0] > 0.5 else "Not Cat"
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
