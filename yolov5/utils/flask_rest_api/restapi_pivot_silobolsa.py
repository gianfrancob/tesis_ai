"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import subprocess
import torch
from PIL import Image
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

IMG_DETECTION_URL = "/v1/img-object-detection/yolov5"
BULK_IMG_DETECTION_URL = "/v1/bulk-img-object-detection/yolov5"
VIDEO_DETECTION_URL = "/v1/video-object-detection/yolov5" # for webcam stream

@app.route(IMG_DETECTION_URL, methods=["POST"])
def predictImage():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img_name = str(image_file).split("FileStorage: '")[1].split("'")[0]
        print("Load file: ", img_name ) # TODO: Printear info de la imagen y del modelo
        save_path = f'./utils/flask_rest_api/postedImages/{img_name}'
        img = Image.open(io.BytesIO(image_bytes))
        img.save(save_path)
        
        logs = subprocess.run(["python3", "detect.py", "--weights", "yolov5s-best.pt", "--source", save_path, "--conf-thres", "0.65", "--augment", "--project", "runs/RESTapi", "--name", "results"])
        
        detected_img_path = f"./runs/RESTapi/results/{img_name}"
        detected_img = Image.open(detected_img_path)
        output = io.BytesIO()
        detected_img.save(output, format='JPEG')
        image_binary = output.getvalue()
        
        #subprocess.run(["rm", save_path])
        
        return send_file(io.BytesIO(image_binary), mimetype='image/jpeg', as_attachment=True, attachment_filename=f'detected_{img_name}')

# TODO: armar predictBulkImage
def predictImageBulk():
    if not request.method == "POST":
        return

    if request.files.get("file"):
        image_file = request.files["file"]
        # TODO: Descomprimir archivo adjuntado
        image_bytes = image_file.read()
        path = str(image_file).split("FileStorage: '")[1].split("'")[0]
        print("Load files: ", path.count() ) # TODO: Printear info de la imagen y del modelo
        save_path = f'./utils/flask_rest_api/postedImages/{img_name}'
        img = Image.open(io.BytesIO(image_bytes))
        img.save(save_path)
        
        logs = subprocess.run(["python3", "detect.py", "--weights", "yolov5s-best.pt", "--source", save_path, "--conf-thres", "0.65", "--augment", "--project", "runs/RESTapi", "--name", "results"])
        
        detected_img_path = f"./runs/RESTapi/results/{img_name}"
        detected_img = Image.open(detected_img_path)
        output = io.BytesIO()
        detected_img.save(output, format='JPEG')
        image_binary = output.getvalue()
        
        #subprocess.run(["rm", save_path])
        
        return send_file(io.BytesIO(image_binary), mimetype='image/jpeg', as_attachment=True, attachment_filename=f'detected_{img_name}') # TODO: ver de eliminar el mimetype
# TODO: armar predictVideo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
