# Step 2: Flask Application
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load the trained model
model_path = './mymodel.keras'
model = load_model(model_path)

# Class Names
class_names = ['LEMON', 'MELON']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = image.load_img(io.BytesIO(file.read()), target_size=(120, 120))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
          
            pred_class = class_names[ (predictions[0][0] > 0.5).astype("int32")]
         
            return render_template('index.html', result=f"{pred_class}")

    return render_template('index.html', result=None)

@app.route('/index.html')
def index():
    return html_template

if __name__ == '__main__':
    app.run(debug=True)
    