#!/usr/bin/env python3

from os.path import join as pjoin
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pedestrian_detection.helper import ROS_INFO, cvbridge
import rospy

from std_msgs.msg import Float32
from pedestrian_detection.msg import LanePose

from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField, CompressedImage
import numpy as np
import cv2


def get_timestamp():
    from datetime import datetime
    now = datetime.now()
    timestring = now.strftime("%m%d%Y-%H%M%S")
    return timestring

class ImageCollector:
    def __init__(self, duckie_name, img_dir):
        self.img_dir = img_dir
        rospy.init_node('image_collector')
        # TODO: fix it to handle general duckie names. Is there rosparam for it??
        image_topic = f'/{duckie_name}/camera_node/image/compressed'  # rospy.get_param("~joy_topic", '/joy_teleop/joy')
        self.img_sub = rospy.Subscriber(image_topic, CompressedImage, self.img_callback, queue_size=1)
        self._recent_timestr = None


    def img_callback(self, compr_img_msg):
        timestring = get_timestamp()
        if timestring == self._recent_timestr:
            return

        _img = cvbridge.compressed_imgmsg_to_cv2(compr_img_msg, desired_encoding="bgr8")

        # Resize image to (112 x 112) and normalize
        resized_img = cv2.resize(_img.copy(), dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        # resized_img = cv2.cvtColor(resized_img.copy(), cv2.COLOR_BGR2RGB)

        # (height, width, channel) -> (channel, width, height)  (?)
        # img = img.transpose(2, 0, 1)

        # Save image
        path = pjoin(self.img_dir, f'frame-{timestring}.png')
        ROS_INFO(f'saving frame to {path}')
        cv2.imwrite(path, _img)



if __name__ == '__main__':
    import os
    import pathlib
    img_dir = '/imgs'
    assert os.path.isdir(img_dir)

    duckie_name = 'yoneduckie'
    collector = ImageCollector(duckie_name, img_dir=img_dir)
    rospy.spin()
