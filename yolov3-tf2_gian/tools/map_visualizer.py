import time
import json
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from mean_average_precision import MeanAveragePrecision

flags.DEFINE_string('classes', './data/test/ps.names', 'path to classes file')
flags.DEFINE_string('path', None,
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
        'output': []
    }

    with open(FLAGS.path, "r") as logFile:
        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt = np.array([
            [439, 157, 556, 241, 0, 0, 0],
            [437, 246, 518, 351, 0, 0, 0],
            [515, 306, 595, 375, 0, 0, 0],
            [407, 386, 531, 476, 0, 0, 0],
            [544, 419, 621, 476, 0, 0, 0],
            [609, 297, 636, 392, 0, 0, 0]
        ])

        # [xmin, ymin, xmax, ymax, class_id, confidence]
        preds = np.array([
            [429, 219, 528, 247, 0, 0.460851],
            [433, 260, 506, 336, 0, 0.269833],
            [518, 314, 603, 369, 0, 0.462608],
            [592, 310, 634, 388, 0, 0.298196],
            [403, 384, 517, 461, 0, 0.382881],
            [405, 429, 519, 470, 0, 0.369369],
            [433, 272, 499, 341, 0, 0.272826],
            [413, 390, 515, 459, 0, 0.619459]
        ])

        # create metric_fn
        metric_fn = MeanAveragePrecision(num_classes=1)

        # add some samples to evaluation
        for i in range(10):
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
