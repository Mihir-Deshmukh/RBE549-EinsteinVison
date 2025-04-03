"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
import json

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray
    print('orient', orient*180/np.pi + 180)

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"


    # img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    img_path = image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()
        
    print('Found %s images'%len(ids))
    with open("/home/mpdeshmukh/RBE549-EinsteinVison/model_outputs/detic/scene_10/detections_merged.json", "r") as file:
        results = json.load(file)

    results_dict = {int(frame_data['frame']): frame_data['objects'] for frame_data in results}

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        frame_id = img_id.split('_')[1]
        result = results_dict[int(frame_id)]

        # detections = yolo.detect(yolo_img)

        # for detection in detections:
        for detection in result:
            
            # if detection["type"] == "car_(automobile)" or detection["type"] == "truck" or detection["type"] == "bus_(vehicle)" or detection["type"] == "pickup_truck" or detection["type"] == "police_cruiser" or detection["type"] == "fire_truck" or detection["type"] == "ambulance" or detection["type"] == "cab_(taxi)":
            if detection["type"] == "car" or detection["type"] == "SUV" or detection["type"] == "sedan" or detection["type"] == "pickup_truck" or detection["type"] == "truck" or detection["type"] == "hatchback":
                detected_class = "car"
            elif detection["type"] == "bicycle" or detection["type"] == "motorcycle":
                detected_class = "car"
            else:
                continue
            
            print('Detected class: %s'%detection["type"])
            # if not averages.recognized_class(detection.detected_class):
            #     continue
            bbox = detection["bbx"]
            # bbox type: {
                #     "x": 433.5802371541502,
                #     "y": 477.302766798419,
                #     "w": 45.0,
                #     "h": 60.0
                # }
            #xywh to xyxy
            
            x_min = bbox["x"] - bbox["w"]/2 
            x_max = bbox["x"] + bbox["w"]/2
            y_min = bbox["y"] - bbox["h"]/2
            y_max = bbox["y"] + bbox["h"]/2
            
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
                
            if x_max > 1279:
                x_max = 1279
            if y_max > 959:
                y_max = 959
            box_2d = [(x_min, y_min), (x_max, y_max)]
            box_2d_int = [
                (int(box_2d[0][0]), int(box_2d[0][1])), 
                (int(box_2d[1][0]), int(box_2d[1][1]))
            ]
            
            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detected_class, box_2d_int, calib_file)
            except:
                print('Invalid 2D Bounding Box')
                print(box_2d_int)
                detection["yaw"] = 90
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = box_2d_int
            detected_class = detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            orient = alpha + theta_ray
            yaw = orient*180/np.pi + 180
            # yaw = (yaw // 15) * 15
            detection["yaw"] = yaw
            
            if not FLAGS.hide_debug:
                print('Estimated pose: %s'%location)

        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            # cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            # cv2.imshow('3D detections', img)
            cv2.imwrite('output/' + img_id + '.png', img)

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(result), time.time() - start_time))
            print('-------------')

        # if FLAGS.video:
        #     cv2.waitKey(1)
        # else:
        #     if cv2.waitKey(0) != 32: # space bar
        #         exit()
        # break
    
    # make updated jsonfile and write
    # print(result)
    reconstructed_results = [{"frame": frame, "objects": objects} for frame, objects in results_dict.items()]
    # print(reconstructed_results)
    
    # with open("/home/mpdeshmukh/RBE549-EinsteinVison/model_outputs/detic/scene_10/detections_processed.json", "w") as file:
    #     json.dump(reconstructed_results, file, indent=4)                                                          

if __name__ == '__main__':
    main()
