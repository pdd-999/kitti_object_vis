""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import os
import sys

import cv2
import numpy as np
from pydantic import conbytes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import argparse

import tqdm

import kitti_util as utils

raw_input = input  # Python 3

def check_if_1_hide_in_2(box_1, depth_1, box_2, depth_2):
    """
    Check if object is hidden behind another object
    """
    # Intersection
    top_left = np.maximum(box_1[0], box_2[0]) # [[x, y]]
    bottom_right = np.minimum(box_1[1], box_2[1]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[0].clip(0) * wh[1].clip(0)
    box_1_area = (box_1[1][0] - box_1[0][0]) * (box_1[1][1] - box_1[0][1])

    if intersection / box_1_area > 0.6 and depth_1 > depth_2:
        return True
    else:
        return False

class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        if split == "training":
            self.num_samples = len(os.listdir(self.image_dir))
        elif split == "testing":
            self.num_samples = len(os.listdir(self.image_dir))
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def isexist_pred_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)


def convert_single_img_to_yolov5_format(img, objects, calib, save_path):
    img_height, img_width, _ = img.shape

    with open(save_path, "w") as output_file:
        for obj in objects:
            if obj.type not in ["Car", "Van", "Truck", "Tram"]:
                continue

            box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
            if box3d_pts_2d is None:
                import ipdb; ipdb.set_trace(context=10)
                box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)

            x1 = obj.xmin
            y1 = obj.ymin
            x2 = obj.xmax
            y2 = obj.ymax

            width = x2-x1
            height = y2-y1
            x_center = (x1+x2)/2
            y_center = (y1+y2)/2
            box_2d = [
                x_center/img_width,
                y_center/img_height,
                width/img_width,
                height/img_height
            ]

            box_3d = box3d_pts_2d.flatten()
            box_3d[0::2] /= img_width
            box_3d[1::2] /= img_height

            sub_label = " ".join(map(str, box_2d+list(box_3d)))
            full_label = "0  " + sub_label
            output_file.write(full_label+"\n")


def convert_single_img_to_hrnet_format(img, objects, calib, save_path):
    img_height, img_width, _ = img.shape

    with open(save_path, "w") as output_file:
        for obj in objects:
            if obj.occlusion != 0:
                continue

            if obj.type not in ["Car", "Van", "Truck", "Tram"]:
                continue

            box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
            if box3d_pts_2d is None:
                import ipdb; ipdb.set_trace(context=10)
                box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)

            x1 = obj.xmin
            y1 = obj.ymin
            x2 = obj.xmax
            y2 = obj.ymax

            width = x2-x1
            height = y2-y1
            x_center = (x1+x2)/2
            y_center = (y1+y2)/2
            box_2d = [
                x_center/img_width,
                y_center/img_height,
                width/img_width,
                height/img_height
            ]

            box_3d = box3d_pts_2d.flatten()
            box_3d[0::2] /= img_width
            box_3d[1::2] /= img_height

            sub_label = " ".join(map(str, box_2d+list(box_3d)))
            full_label = "0  " + sub_label
            output_file.write(full_label+"\n")


def kitti2yolov5(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    save_path = os.path.join(root_dir, "training", "yolov5_label")
    os.makedirs(save_path, exist_ok=True)
    for data_idx in tqdm.tqdm(range(len(dataset))):
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx)
        else:
            objects = []

        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)

        convert_single_img_to_yolov5_format(img, objects, calib, os.path.join(save_path,"%06d.txt" % (data_idx)))

def kitti2hrnet(root_dir, args, out_name="hrnet_label"):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    save_path = os.path.join(root_dir, "training", out_name)
    os.makedirs(save_path, exist_ok=True)
    for data_idx in tqdm.tqdm(range(len(dataset))):
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx)
        else:
            objects = []

        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)

        convert_single_img_to_hrnet_format(img, objects, calib, os.path.join(save_path,"%06d.txt" % (data_idx)))

def visualize_single_data_point(idx):
    base_path = "KITTI/training"
    img_path = os.path.join(base_path, "image_2", "%06d.png" % (idx))
    label_path = os.path.join(base_path, "yolov5_label", "%06d.txt" % (idx))
    img = cv2.imread(img_path)
    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()
    
    img = cv2.imread(img_path)
    img_2d = img.copy()
    img_3d = img.copy()
    img_height, img_width, _ = img.shape
    for line in lines:
        data = line.split(' ')
        data = np.array(list(map(float, data[2:])))
        landmarks = data[4:]

        x_center, y_center, width, height = data[0:4]
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        img_2d = cv2.rectangle(img_2d, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)

        landmarks[0::2] *= img_width
        landmarks[1::2] *= img_height
        img_3d = utils.draw_projected_box3d(img_3d, landmarks.reshape(8,2).astype(int))

    cv2.imwrite("img_2d.png", img_2d)
    cv2.imwrite("img_3d.png", img_3d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="data/object",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "--viz",
        type=int,
        default=-1
    )
    args = parser.parse_args()

    if args.viz != -1:
        visualize_single_data_point(args.viz)
    else:
        # kitti2yolov5(args.dir, args)
        kitti2hrnet(args.dir, args)
