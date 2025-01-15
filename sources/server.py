from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import server_utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_infer = server_utils.ModelInfer()

@app.route('/', methods=['GET', 'POST'])
def index():
    captions = []
    uploaded_image_path = None

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = secure_filename(image.filename)
                uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(uploaded_image_path)

        language = request.form.get('language')

        if uploaded_image_path:
            if language == 'romanian':
                captions = model_infer.caption_image_romanian(uploaded_image_path)
            elif language == 'english':
                captions = model_infer.caption_image_english(uploaded_image_path)

    return render_template('index.html',
                           captions=captions,
                           image_path=uploaded_image_path)

if __name__ == '__main__':
    app.run(debug=True)
