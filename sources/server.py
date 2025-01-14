from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads' # r'C:\Users\Vlad\ibd_image_captioning\sources\static\uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dummy model inference functions (replace with actual inference logic)
def caption_image_romanian(image_path):
    return [f"Romanian caption by Model {i+1}: [Caption for {os.path.basename(image_path)}]" for i in range(2)]

def caption_image_english(image_path):
    return [f"English caption by Model {i+1}: [Caption for {os.path.basename(image_path)}]" for i in range(4)]

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    captions = []
    uploaded_image_path = None

    if request.method == 'POST':
        # Handle file upload
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = secure_filename(image.filename)
                uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(uploaded_image_path)

        # Get selected language
        language = request.form.get('language')

        # Generate captions for all models in the selected language
        if uploaded_image_path:
            if language == 'romanian':
                captions = caption_image_romanian(uploaded_image_path)
            elif language == 'english':
                captions = caption_image_english(uploaded_image_path)

    return render_template('index.html',
                           captions=captions,
                           image_path=uploaded_image_path)

if __name__ == '__main__':
    # with open(r'C:\Users\Vlad\ibd_image_captioning\sources\templates\index.html') as f:
    #     text = f.read()
    #     print(text)

    app.run(debug=True)
