from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
import os
import tensorflow as tf
from utils import get_pose_data

FPS = 14
DURATION = 2.5

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files

class UploadForm(FlaskForm):
    file = FileField('Upload Video', validators=[
        FileRequired(),
        FileAllowed(['mp4'], 'Allowed file types: mp4')
    ])

@app.route('/', methods=['GET', 'POST'])
def classify():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = file.filename
        # Create the directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        loaded_model = tf.keras.models.load_model('fall_detection.h5')
        keypoints = get_pose_data(FPS, DURATION, app.config['UPLOAD_FOLDER'] + '/' + filename, False, False)
        print (app.config['UPLOAD_FOLDER'] + filename)
        pose_list = []
        pose_list.append(tf.convert_to_tensor(keypoints))
        pose_data = tf.stack(pose_list)
        # Use the loaded model for predictions on new data
        predictions = loaded_model.predict(pose_data)
        binary_predictions = ['Fall' if pred > 0.5 else 'Not Fall' for pred in predictions]
        # Process the predictions as needed
        return 'Video {} classified as {}'.format(filename, binary_predictions)
        
    return render_template('upload.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
