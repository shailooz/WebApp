from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import gc  # Garbage collection

app = Flask(__name__)

# Load the trained model once at the start
model_path = './mymodel.keras'
model = load_model(model_path)

# Class Names
class_names = ['LEMON', 'MELON']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                # Load and preprocess the image
                img = image.load_img(io.BytesIO(file.read()), target_size=(120, 120))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict class
                predictions = model.predict(img_array)
                pred_class = class_names[int(predictions[0][0] > 0.5)]

                # Force garbage collection to free memory
                del img, img_array, predictions
                gc.collect()

                return render_template('index.html', result=f"{pred_class}")
            except Exception as e:
                return f"Error processing the image: {str(e)}"

    return render_template('index.html', result=None)

@app.route('/index.html')
def index():
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
