import time
import json
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from mean_average_precision import MeanAveragePrecision
import functools 

flags.DEFINE_string('classes', './data/test/ps.names', 'path to classes file')
flags.DEFINE_string('path', './tools/logs.json',
                    'path to json file with ground truth and detections')
flags.DEFINE_integer('size', 416, 'images size')
flags.DEFINE_string('outputLogPath', './tools/map.log.json', 'path to output log file')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')


"""
To run this:

/workspace/shared_volume/tesis_ai/yolov3-tf2_gian# python ./tools/map_visualizer.py --path ./logs...json
"""


def getCoords(pred, size, logs, ctype):
    # size = 1 
    pred_cp = pred

    left = pred_cp[0] * size
    top = pred_cp[1] * size
    right = pred_cp[2] * size
    bottom = pred_cp[3] * size
    # if (ctype == "pd"):
    #     left = pred_cp[0] * size
    #     top = pred_cp[1] * size
    #     right = (pred_cp[2]) * size
    #     bottom = (pred_cp[3]) * size
    # elif (ctype == "gt"):
    #     left = pred_cp[0] * size
    #     top = pred_cp[1] * size
    #     right = pred_cp[2] * size
    #     bottom = pred_cp[3] * size
    # else:
    #     raise Exception("Coordinate type not supported")
    newPred = [left, top, right, bottom]
    logs["getCoords"] += [{
        "coor_type": ctype,
        "size": size,
        "coords": pred,
        "newcds": newPred
    }]

    return newPred

def main(_argv):
    logs = {
        'classes': FLAGS.classes,
        'path': FLAGS.path,
        'img_size': FLAGS.size,
        'num_classes': FLAGS.num_classes,
        'output': {},
        'getCoords': []
    }
    with open(FLAGS.classes, "r") as classesFile:
        classes = classesFile.readlines()
        classesFile.close()

    preds = []
    gt = []
    with open(FLAGS.path, "r") as logFile:
        data = json.load(logFile)
        # Iterate over predictions
        for out in data["output"]:
            groundTruth = out["true_detections"]
            rawPreds = out["detections"]
            
            # PREDICTIONS
            for p in rawPreds:
                # Skip if no prediction
                if (type(p) == type({})):
                    # Map class name to class idx
                    predClass = None
                    for i in range(len(classes)):
                        if (str(p["class_name"]).strip() == str(classes[i]).strip()):
                            predClass = i
                            break
                    # Build prediction vector formatted in a way mAP could use it
                    if (predClass != None):
                        boxes = getCoords(p["boxes"], data["img_resize"], logs, "pd")
                        preds += [boxes + [predClass, p["score"]]]

            # GROUND TRUTH
            ctr = 0
            for i in range(len(groundTruth)):
                idx = i - ctr
                 # Build groun truth vector formatted in a way mAP could use it
                label = groundTruth[idx][:4]
                groundTruth[idx][:4] = getCoords(label, data["img_resize"], logs, "gt")
                groundTruth[idx] += [0, 0]

                # Delete vectors full of zeros
                if (functools.reduce(lambda a,b : a+b, label[0:3]) == 0):
                    del groundTruth[idx]
                    ctr += 1
            gt += groundTruth
        

        logs["output"]["GROUND TRUTH"] = gt
        logs["output"]["PREDS"] = preds

        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt = np.array(gt)

        # [xmin, ymin, xmax, ymax, class_id, confidence]
        preds = np.array(preds)

        # TODO: chequear gt y preds a ver xq son de dimension 0 para los logs de stock weights
        '''
        (yolov3-tf2-gpu) root@727c9f86d38e:/workspace/shared_volume/tesis_ai/yolov3-tf2_gian# python tools/map_visualizer.py  --classes ./data/test/ps.names --path ./output/pivot_silobolsa/stock_weights/logs_2021-01-07\ 23\:52\:34.849503.json --outputLogPath ./output/pivot_silobolsa/stock_weights/
Traceback (most recent call last):
  File "tools/map_visualizer.py", line 151, in <module>
    app.run(main)
  File "/root/anaconda3/envs/yolov3-tf2-gpu/lib/python3.7/site-packages/absl/app.py", line 300, in run
    _run_main(main, args)
  File "/root/anaconda3/envs/yolov3-tf2-gpu/lib/python3.7/site-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "tools/map_visualizer.py", line 126, in main
    metric_fn.add(preds, gt)
  File "/root/anaconda3/envs/yolov3-tf2-gpu/lib/python3.7/site-packages/mean_average_precision/mean_average_precision.py", line 63, in add
    match_table = compute_match_table(preds_c, gt_c, self.imgs_counter)
  File "/root/anaconda3/envs/yolov3-tf2-gpu/lib/python3.7/site-packages/mean_average_precision/utils.py", line 139, in compute_match_table
    difficult = np.repeat(gt[:, 5], preds.shape[0], axis=0).reshape(preds[:, 5].shape[0], -1).tolist()
ValueError: cannot reshape array of size 0 into shape (0,newaxis)

        '''

        np.savetxt('./tools/mapLogs_gt.csv', gt, delimiter=',')
        np.savetxt('./tools/mapLogs_preds.csv', preds, delimiter=',')


        # create metric_fn
        metric_fn = MeanAveragePrecision(num_classes=FLAGS.num_classes)

        # add some samples to evaluation
        for i in range(len(data["output"])):
            metric_fn.add(preds, gt)

        # compute PASCAL VOC metric
        aux = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
        print("VOC PASCAL mAP: ", aux)
        logs["output"]["VOC PASCAL mAP"] = str(aux)

        # compute PASCAL VOC metric at the all points
        aux = metric_fn.value(iou_thresholds=0.5)['mAP']
        print("VOC PASCAL mAP in all points: ", aux)
        logs["output"]["VOC PASCAL mAP in all points"] = str(aux)

        # compute metric COCO metric
        aux = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        print("COCO mAP: ", aux)
        logs["output"]["COCO mAP"] = str(aux)

        with open(FLAGS.outputLogPath + "/map.log.json", "w") as f:
            json.dump(logs, f, indent=2)

        logFile.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass







