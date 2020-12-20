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
flags.DEFINE_string('output', './output', 'path to output image')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')


"""
To run this:

/workspace/shared_volume/tesis_ai/yolov3-tf2_gian# python ./tools/map_visualizer.py --path ./logs...json
"""

def main(_argv):
    logs = {
        'classes': FLAGS.classes,
        'path': FLAGS.path,
        'img_size': FLAGS.size,
        'num_classes': FLAGS.num_classes,
        'output': ""
    }
    with open(FLAGS.classes, "r") as classesFile:
        classes = classesFile.readlines()
        classesFile.close()

    preds = []
    gt = []
    with open(FLAGS.path, "r") as logFile:
        data = json.load(logFile)
        for out in data["output"]:
            groundTruth = out["true_detections"]
            rawPreds = out["detections"]
            for p in rawPreds:
                if (type(p) == type({})):
                    predClass = None
                    for i in range(len(classes)):
                        if (str(p["class_name"]).strip() == str(classes[i]).strip()):
                            predClass = i
                            break
                    if (predClass != None):
                        preds += [p["boxes"] + [predClass, p["score"]]]

            ctr = 0
            for i in range(len(groundTruth)):
                idx = i - ctr
                label = groundTruth[idx]
                for j in range(len(label[:-1])):
                    groundTruth[idx][j] = label[j] * data["img_resize"]
                groundTruth[idx] += [0, 0]
                
                if (functools.reduce(lambda a,b : a+b, label[:-1]) == 0):
                    del groundTruth[idx]
                    ctr += 1


            for i in range(len(preds)):
                label = preds[i]
                for j in range(len(label[:-2])):
                    preds[i][j] = label[j] * data["img_resize"]

            logs["output"] += "\n" + str("groundTruth: " + str(groundTruth))
            logs["output"] += "\n" + str("GT: " + str(gt))
            gt += groundTruth

                    
        logFile.close()
            

        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt = np.array(gt)

        # [xmin, ymin, xmax, ymax, class_id, confidence]
        preds = np.array(preds)

        logs["output"] += "\n" + str("GROUND TRUTH: " + np.array_str(gt))
        logs["output"] += "\n" + str("PREDS: " + np.array_str(preds))

        np.savetxt('./tools/mapLogs_gt.csv', gt, delimiter=',')
        np.savetxt('./tools/mapLogs_preds.csv', preds, delimiter=',')

        with open("./tools/mapLogs.json", "w") as f:
            f.write(logs["output"])

        # create metric_fn
        metric_fn = MeanAveragePrecision(num_classes=FLAGS.num_classes)

        # add some samples to evaluation
        for i in range(len(logs["output"])):
            metric_fn.add(preds, gt)

        # compute PASCAL VOC metric
        print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

        # compute PASCAL VOC metric at the all points
        print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

        # compute metric COCO metric
        print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass





