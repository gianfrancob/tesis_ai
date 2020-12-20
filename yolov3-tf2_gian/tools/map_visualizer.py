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


    with open(FLAGS.path, "r") as logFile:
        data = json.load(logFile)
        for out in data["output"]:
            groundTruth = out["true_detections"]
            rawPreds = out["detections"]
            preds = []
            for p in rawPreds:
                if (type(p) == type({})):
                    predClass = None
                    for i in range(len(classes)):
                        logs["output"] += str(p["class_name"])
                        logs["output"] += str(classes[i])
                        if (id(p["class_name"]) is id(classes[i])):
                            logs["output"] += str("IGUALES!!!!!!!!!!!!!!!")
                            predClass = i
                            break
                    if (predClass != None):
                        logs["output"] += "\n" + str("RAW PRED BOXES: " + json.dumps(p["boxes"]))
                        logs["output"] += "\n" + str("RAW PRED predClass: " + predClass)
                        preds += p["boxes"].append(predClass)
                    

            for label in groundTruth + preds:
                # logs["output"] += "\n" + str("GROUD TRUTH BOXES: " + json.dumps(label))
                if (functools.reduce(lambda a,b : a+b, label[:-1]) == 0):
                    # logs["output"] += "\n" + str("deleting...")
                    del label
                else:
                    for i in range(len(label[:-1])):
                        label[i] = label[i] * data["img_resize"]
                    logs["output"] += "\n" + str("GROUD TRUTH BOXES AFTER: " + json.dumps(label))
                    
        logFile.close()
            

        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt = np.array(groundTruth)

        # [xmin, ymin, xmax, ymax, class_id, confidence]
        preds = np.array(preds)

        logs["output"] += "\n" + str("GROUD TRUTH: " + np.array_str(gt))
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


