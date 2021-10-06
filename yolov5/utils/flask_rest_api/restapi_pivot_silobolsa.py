"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import subprocess
import torch
import base64
import json
import re
import os
import pyunpack
from PIL import Image
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

IMG_DETECTION_URL = "/v1/img-object-detection/yolov5"
BULK_IMG_DETECTION_URL = "/v1/bulk-img-object-detection/yolov5"
VIDEO_DETECTION_URL = "/v1/video-object-detection/yolov5" # for webcam stream

@app.route(IMG_DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return app.response_class(
            status=400, 
            mimetype='application/json',
            response="Method must be POST"
        )

    try: 
        if request.files.get("attachment"):
            attached_file = request.files["attachment"]
            attached_file_str = str(attached_file)
            print("attached_file ", attached_file_str)

            extension = ""
            if len(re.findall("application/", attached_file_str)) > 0:
                extension = "compressed"
            elif len(re.findall("image/", attached_file_str)) > 0:
                extension = "image"

            print("attached_file extension ", extension)

            if extension == "":
                return app.response_class(
                    status=400, 
                    mimetype='application/json',
                    response="Bad attachment extension. Supported types: Check https://github.com/ponty/pyunpack/tree/0.2.2"
                )
            else:
                attachment_name = attached_file.filename.split("\\")[-1]
                # attachment_bytes = attached_file.read()
                print("Load file: ", attachment_name )
                save_path = f'./utils/flask_rest_api/postedImages/{attachment_name}'
                attached_file.save(save_path)
                detected_img_path = f"./runs/RESTapi/results/{attachment_name}"
                sub_results_folder = ""
                if extension == "compressed":
                    print("HOLA")
                    save_dir = "." + save_path.split(".")[1]
                    mkdir_logs = subprocess.run(["mkdir", save_dir], stdout=subprocess.PIPE, text=True).stdout        
                    pyunpack.Archive(save_path).extractall(save_dir)
                    save_path = save_dir
                    sub_results_folder = "/" + attachment_name
                    detected_img_path = f"./runs/RESTapi/results{sub_results_folder}"
                    print("detected_img_path: ", detected_img_path)
            
            print("save_path ", save_path)
            detect_logs = subprocess.run(["python3", "detect.py", "--weights", "yolov5s-best.pt", "--source", save_path, "--conf-thres", "0.65", "--augment", "--project", "runs/RESTapi", "--name", f"results{sub_results_folder}"], stdout=subprocess.PIPE, text=True).stdout
            print(detect_logs)

            output = io.BytesIO()
            inference_binary = None
            if extension == "image":
                detected_img = Image.open(detected_img_path)
                detected_img.save(output, format='PNG')
                inference_binary = output.getvalue()
                output.seek(0)
            else:
                output_compressed_file = detected_img_path + "/output.tar.gz"
                wd = os.getcwd()
                os.chdir(detected_img_path)
                os.system("ls -l")
                os.system("tar vczf output.tar.gz *")
                os.system("ls -l")
                os.chdir(wd)

                with open(output_compressed_file, 'rb') as file_data:
                    output = file_data.read()
                    inference_binary = output # TODO: chequear en FE si sirve lo que le llega

            
            #subprocess.run(["rm", save_path]) TODO:: no colgar con esto, para no llenar la matrix de basoooooooura
            
            # TODO: Rearmar logs cuadno extension != image
            inference_logs = detect_logs.split(': ')[1].split(', Done')[0] # 416x640 5 pivots
            detections = {}
            pivots = re.findall("[0-9]+ pivot", inference_logs)
            if len(pivots) > 0:
                detections['pivots'] = pivots[0].split(' ')[0]
            silobolsas = re.findall("[0-9]+ silobolsa", inference_logs)
            if len(silobolsas) > 0:
                detections['silobolsas'] = silobolsas[0].split(' ')[0]

            input_img_size = detect_logs.split('imgsz=')[1].split(',')[0]
            input_img_size += "x" + input_img_size + " px"

            logs = dict(
                image_name=attachment_name,
                input_img_size=input_img_size,
                img_size = inference_logs.split(' ')[0] + " px",
                conf_thres = detect_logs.split('conf_thres=')[1].split(',')[0],
                iou_thres = detect_logs.split('iou_thres=')[1].split(',')[0],
                inference_time = detect_logs.split('Done. (')[1].split(')')[0],
                total_time = detect_logs.split('Done. (')[2].split(')')[0],
                detections = detections,
                raw_logs=detect_logs
            )
            print(json.dumps(logs, indent=4))
            
            
            data = dict(
                inference=base64.encodebytes(inference_binary).decode('ascii'),
                logs=logs
            )
            
            response = app.response_class(
                response=json.dumps(data),
                status=200,
                mimetype='application/json'
            )
            return response
    except Exception as e:
        print(f"Error! {e}")
        return  app.response_class(
            status=400, 
            mimetype='application/json',
            response=f"Detection failed. Error: {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
