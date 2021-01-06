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







