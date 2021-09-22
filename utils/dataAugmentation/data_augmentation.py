
# Importing Libraries
import os
import numpy as np
import cv2
import argparse
import time
from tqdm import tqdm


stage = "train"
# stage = "val"

claseName = "Pivot"
# claseName = "Silobolsa"

PATH = f'/home/gianfranco/workspace/shared_volume/images_bolsasPivotAug/{stage}/Images/' + claseName

#convert from Yolo_mark to opencv format
def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H
    voc = []
    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))
    return [int(v) for v in voc]

# Convert from opencv format to yolo format
# H,W is the image height and width
def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]
    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2
    return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H, 6)

def fakeCvFormattoYolo(corner, H, W):
    return corner[0], corner[1], corner[2], corner[3], corner[4]

class yoloRotatebbox:
    def __init__(self, filename, image_ext, angle):
        label_filename = PATH + "/clf/" + filename
        label_filename = label_filename.replace("Images", "Labels")

        # print("-------------- filename: ", PATH + "/" + filename + image_ext)
        # print("-------------- label_filename: ", label_filename + '.txt')
        assert os.path.isfile(PATH + "/" + filename + image_ext)
        assert os.path.isfile(label_filename + '.txt')
        
        self.filename = PATH + "/" + filename
        self.image_ext = image_ext
        self.label = label_filename
        self.angle = angle
        
        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)
        
        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotateYolobbox(self):
        new_height, new_width = self.rotate_image().shape[:2]
        f = open(self.label + '.txt', 'r')
        f1 = f.readlines()
        new_bbox = []
        H, W = self.image.shape[:2]
        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]) #yoloFormattocv(float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), H, W)
                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)
                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []
                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift, lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime
                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)
                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1], new_lower_right_corner[0], new_lower_right_corner[1]])
        return new_bbox
    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]
        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat

if __name__ == "__main__":
    angels=[45,90,135,180,225,270,315]

    for filename in tqdm(sorted(os.listdir(PATH))):
        file =filename.split(".")
        if(file[-1]=="JPG"):
            image_name=file[0]
            image_ext="."+file[1]
        else:
            continue
        for angle in angels:
            im = yoloRotatebbox(image_name, image_ext, angle)
            bbox = im.rotateYolobbox()
            image = im.rotate_image()
            # to write rotateed image to disk
            cv2.imwrite(PATH + "/" + image_name+'_' + str(angle) + '.JPG', image)
            label_filename = PATH + "/clf/" + image_name +'_' + str(angle) + '.txt'
            label_filename = label_filename.replace("Images", "Labels")
            #print("For angle "+str(angle))
            if os.path.exists(label_filename):
                os.remove(label_filename)
            # to write the new rotated bboxes to file
            for i in bbox:
                with open(label_filename, 'a') as fout:
                    fout.writelines(
                        ' '.join(map(str, fakeCvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')
                        # ' '.join(map(str, cvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')