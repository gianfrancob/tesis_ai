import sys

sys.path.append('./')

import time
import json
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/test/test.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', "None", 'path to input image')
flags.DEFINE_string('tfrecord', './data/test/val2.tfrecord', 'tfrecord instead of image')
flags.DEFINE_boolean('shuffle', False, 'tfrecord shuffle')
flags.DEFINE_string('output', './output', 'path to output image')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')


"""
To run this:
Ex:
cd /workspace/shared_volume/tesis_ai/yolov3-tf2_gian
python ./tools/bulk_detector.py --weights ./checkpoints/saved_weights/yolov3_train_12.tf
"""

def main(_argv):
    logs = {
        'classes': FLAGS.classes,
        'weights': FLAGS.weights,
        'tiny': FLAGS.tiny,
        'img_resize': FLAGS.size,
        'image': FLAGS.image,
        'tfrecord': FLAGS.tfrecord,
        'shuffle': FLAGS.shuffle,
        'num_classes': FLAGS.num_classes,
        'output': []
    }

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    images_raw = []
    labels = []
    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        logging.info("Dataset loaded!")
        if FLAGS.shuffle:
            dataset = dataset.shuffle(512)
        for item in dataset:
            image_raw, label = item
            images_raw += [image_raw]
            labels += [np.array(label).tolist()]
        
        # for i in range(len(labels)):
        #     logging.info(f"LABEL[{i}]: \n {labels[i]} \n")
    else:
        images_raw = [tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)]

    img_idx = 0
    total_time = 0
    for img_raw in images_raw:
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        elapsed_time = t2 - t1
        logging.info('time: {}'.format(elapsed_time))
        total_time += elapsed_time

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        output_name = f'{FLAGS.output}/detected_img_{img_idx}.jpg'

        logging.info('detections:')
        logs["output"] += [{
            "idx": img_idx,
            "path": output_name,
            "time": elapsed_time,
            "detections": [],
            "true_detections": labels[img_idx]
        }]

        for i in range(nums[0]):
            _class_name = class_names[int(classes[0][i])]
            _score = np.array(scores[0][i]).tolist()
            _boxes = np.array(boxes[0][i]).tolist()
            logging.info('\t{}, {}, {}'.format(_class_name, _score, _boxes))
            logs["output"][img_idx]["detections"] += [{
                "class_name": _class_name,
                "score": _score,
                "boxes": _boxes
            }]
        
        
        cv2.imwrite(output_name, img)
        logging.info('output saved to: {}'.format(output_name))

        img_idx += 1
    
    logs["total_time"] = total_time
    logs["avg_time"] = total_time / img_idx

    
    logging.info(json.dumps(logs, indent=2))
    with open(f'{FLAGS.output}/logs_{datetime.now()}.json', 'w') as json_file:
        json.dump(logs, json_file)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
