import time
import json
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
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', './data/test/train.tfrecord', 'tfrecord instead of image')
flags.DEFINE_string('shuffle', False, 'tfrecord shuffle')
flags.DEFINE_string('output', './output', 'path to output image')
flags.DEFINE_integer('num_classes', 2, 'number of classes in the model')


def main(_argv):
    logs = {
        'classes': flags.classes,
        'weights': flags.weights,
        'tiny': flags.tiny,
        'img_resize': flags.size,
        'image': flags.image,
        'tfrecord': flags.tfrecord,
        'shuffle': flags.shuffle,
        'num_classes': flags.num_classes,
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
    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        if FLAGS.shuffle:
            dataset = dataset.shuffle(512)
        for i in dataset:
            image_raw, _label = next(iter(dataset.take(1)))
            images_raw += (image_raw, _label)
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
        logs["output"] += {
            "idx": img_idx,
            "path": output_name,
            "time": elapsed_time,
            "detections": []
        }

        for i in range(nums[0]):
            class_name = class_names[int(classes[0][i])]
            score = np.array(scores[0][i])
            boxes = np.array(boxes[0][i]))
            logging.info('\t{}, {}, {}'.format(class_name, score, boxes)
            logs["output"][img_idx]["detections"] += {
                "class_name": class_name,
                "score": score,
                "boxes": boxes
            }
        
        
        cv2.imwrite(output_name, img)
        logging.info('output saved to: {}'.format(output_name))

        img_idx += 1
    
    logs["total_time"] = total_time
    logs["avg_time"] = total_time / img_idx

    print(json.dump(logs))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
