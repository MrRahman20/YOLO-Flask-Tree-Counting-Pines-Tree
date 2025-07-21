from flask import Flask, render_template, request, redirect, send_from_directory
from model_processing import process_image
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_files = request.files.getlist('image_files')
        chm_files = request.files.getlist('chm_files')
        
        # Get filter values
        chm_filter = float(request.form.get('chm_filter', 2.5))
        confidence = float(request.form.get('confidence', 0.05))
        iou = float(request.form.get('iou', 0.1))
        cluster_eps = float(request.form.get('cluster_eps', 2.86))  # New slider value for clustering

        if not image_files or not chm_files:
            print("No files uploaded")
            return redirect(request.url)

        # Create directories to store the uploaded files temporarily
        image_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'image_files')
        chm_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'chm_files')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(chm_dir, exist_ok=True)

        # Save the uploaded files
        for file in image_files:
            file.save(os.path.join(image_dir, secure_filename(file.filename)))

        for file in chm_files:
            file.save(os.path.join(chm_dir, secure_filename(file.filename)))

        # Extract the main image and CHM files
        image_path = next((os.path.join(image_dir, f.filename) for f in image_files if f.filename.endswith('.png') or f.filename.endswith('.tif')), None)
        chm_path = next((os.path.join(chm_dir, f.filename) for f in chm_files if f.filename.endswith('.tif')), None)

        if not image_path or not chm_path:
            print("Image or CHM file not found")
            return redirect(request.url)

        # Process the image with the YOLO model
        zip_output_path, tree_count, xml_output_path = process_image(image_path, chm_path, chm_filter, confidence, iou, cluster_eps)
        
        # Replace backslashes with forward slashes in zip_output_path
        zip_output_path = zip_output_path.replace('\\', '/')
        xml_output_path = xml_output_path.replace('\\', '/')
        
        print(f"zip_output_path: {zip_output_path}, tree_count: {tree_count}, xml_output_path: {xml_output_path}")
        
        return render_template('index.html', zipfile=zip_output_path, tree_count=tree_count, xmlfile=xml_output_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    try:
        app.run(debug=False, use_reloader=False)  # ✅ aman, gak bikin loop reloader
    except KeyboardInterrupt:
        print("🛑 Flask app dihentikan secara manual.")
