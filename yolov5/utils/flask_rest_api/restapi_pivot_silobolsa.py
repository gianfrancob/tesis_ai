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
                    save_dir = "." + save_path.split(".")[1]
                    mkdir_logs = subprocess.run(["mkdir", save_dir], stdout=subprocess.PIPE, text=True).stdout        
                    pyunpack.Archive(save_path).extractall(save_dir)
                    save_path = save_dir
                    sub_results_folder = "/"# + attachment_name
                    detected_img_path = f"./runs/RESTapi/results{sub_results_folder}"
                    print("detected_img_path: ", detected_img_path)
            
            print("save_path ", save_path)
            detect_logs = subprocess.run(["python3", "detect.py", "--weights", "yolov5s-best.pt", "--source", save_path, "--conf-thres", "0.65", "--augment", "--project", "runs/RESTapi", "--name", f"results{sub_results_folder}"], stdout=subprocess.PIPE, text=True).stdout
            print("detect_logs\n", detect_logs)

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
                    inference_binary = output 

            os.system(f"rm -r ./runs/RESTapi/results/*")
            os.system(f"rm -r ./utils/flask_rest_api/postedImages/*")
            '''
            Namespace(weights=['yolov5s-best.pt'], source='./utils/flask_rest_api/postedImages/bulkImages', imgsz=640, conf_thres=0.65, iou_thres=0.45, max_det=1000, device='', view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=True, update=False, project='runs/RESTapi', name='results/bulkImages.zip', exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False)
            image 1/6 /workspace/shared_volume/yolov5/utils/flask_rest_api/postedImages/bulkImages/bolsa1.png: 320x640 1 silobolsa, Done. (0.050s)
            image 2/6 /workspace/shared_volume/yolov5/utils/flask_rest_api/postedImages/bulkImages/pivot_silo1.png: 352x640 1 pivot, 2 silobolsas, Done. (0.049s)
            image 3/6 /workspace/shared_volume/yolov5/utils/flask_rest_api/postedImages/bulkImages/pivot_silo2.png: 384x640 1 silobolsa, Done. (0.050s)
            image 4/6 /workspace/shared_volume/yolov5/utils/flask_rest_api/postedImages/bulkImages/pivot_silo3.png: 352x640 1 silobolsa, Done. (0.049s)
            image 5/6 /workspace/shared_volume/yolov5/utils/flask_rest_api/postedImages/bulkImages/riego1.png: 416x640 5 pivots, Done. (0.051s)
            image 6/6 /workspace/shared_volume/yolov5/utils/flask_rest_api/postedImages/bulkImages/riego2.png: 512x640 24 pivots, Done. (0.053s)

            '''
            input_img_size = detect_logs.split('imgsz=')[1].split(')')[0]
            input_img_size += "x" + input_img_size + " px"


            parsed_inference_logs = []
            inference_logs = ''.join(detect_logs.split(")")[1:]).split("image ")[1:]
            # for inference in inference_logs:
            for inference in detect_logs.split("image")[1:-1]:
                print("inference ", inference)
                name = inference.split(': ')[0].split('/')[-1]
                inference_row = inference.split(': ')[1]#.split(', Done')[0] # 416x640 5 pivots
                inference_time = inference_row.split('Done. (')[1].split('s')[0] + 's'
                detections = {}
                pivots = re.findall("[0-9]+ pivot", inference_row)
                if len(pivots) > 0:
                    detections['pivots'] = pivots[0].split(' ')[0]
                silobolsas = re.findall("[0-9]+ silobolsa", inference_row)
                if len(silobolsas) > 0:
                    detections['silobolsas'] = silobolsas[0].split(' ')[0]
                
                img_size = inference_row.split(' ')[0] + " px"

                parsed_inference_logs += [dict(
                    image_name=name,
                    img_size = img_size,
                    detections = detections,
                    inference_time = inference_time
                )]

            logs = dict(
                filename = attachment_name,
                inference_logs = parsed_inference_logs,
                input_img_size=input_img_size,
                conf_thres = detect_logs.split('conf_thres=')[1].split(',')[0],
                iou_thres = detect_logs.split('iou_thres=')[1].split(',')[0],
                total_time = detect_logs.split('Speed:')[1].split(', ')[1].split("inference")[0].replace("inference", ""),
                raw_logs=detect_logs
            )
            # print(json.dumps(logs, indent=4))
            
            
            data = dict(
                inference="data:application/zip;base64," + base64.encodebytes(inference_binary).decode('ascii'),
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
