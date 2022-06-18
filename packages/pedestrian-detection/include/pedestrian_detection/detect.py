#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pedestrian_detection.log_helper import ROS_INFO
from cv_bridge import CvBridge
import rospy

from std_msgs.msg import Bool
from pedestrian_detection.msg import LanePose

from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField, CompressedImage
import numpy as np
import cv2

def colorize_segmentation(segm):
    """Returns an image from segmentation result tensor"""
    preset_colors = np.array([
        [84, 142, 41],  # Green: road
        [254, 255, 10],  # Yellow: center line
        [255, 255, 255],  # White: side lines
        [81, 91, 218],  # light blue: duckie
        [251, 0, 255],  # pink: sky
        [217, 89, 82],  # orange: cone (only seldom show up...)
        [0, 0, 0],  # black: anything else
    ], dtype=int)

    ROS_INFO(f'segmentation input shape: {segm.shape}')  # (1, 7, 56, 56)
    segm = segm.squeeze()
    out = preset_colors[segm.argmax(axis=0)]
    ROS_INFO(f'segmentation output shape: {out.shape}')  # (1, 7, 56, 56)
    return out


def load_model(path, device, pedest_only=False):
    from pedestrian_detection.model import make_model
    if pedest_only:
        raise NotImplementedError

    model = make_model().to(device)
    state_dicts = torch.load(path, map_location=device)
    key2module = {'encoder': model.encoder,
                  'pedest_lane': model.pedest.lane_head,
                  'lanepos_offset': model.lanepos.offset_head,
                  'lanepos_phi': model.lanepos.phi_head,
                  'segm': model.segment.segm_head}
    for key, module in key2module.items():
        module.load_state_dict(state_dicts[key])

    return model


# def main(model_path):
#     assert torch.cuda.is_available()

#     # Load the pretrained model
#     ROS_INFO(f'loading pretrained model from {model_path}')
#     model = load_model(model_path).cuda()

#     # Perform detection
#     segm = model.segment.predict(img)
#     offset, phi = model.lanepos.predict(img)
#     _, pedest = model.pedest.predict(img)


class Detector:
    bridge = CvBridge()  # TODO: Can it work fine?
    def __init__(self, duckie_name, model_path) -> None:
        # assert torch.cuda.is_available()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.init_node('detector')

        # Load the pretrained model
        ROS_INFO(f'loading pretrained model from {model_path}...')
        self.model = load_model(model_path, device)
        ROS_INFO(f'loading pretrained model from {model_path}...done')

        # self.segm_pub = rospy.Publisher("/detection/segmentation", queue_size=1)
        self.segm_vis_pub = rospy.Publisher("/detection/segmentation_vis", CompressedImage, queue_size=1)  # Just for visualization
        self.lanepose_pub = rospy.Publisher("/detection/lanepose", LanePose, queue_size=1)
        self.pedest_pub = rospy.Publisher("/detection/pedestrian", Bool, queue_size=1)

        image_topic = f'/{duckie_name}/camera_node/image/compressed'  # rospy.get_param("~joy_topic", '/joy_teleop/joy')
        self.img_sub = rospy.Subscriber(image_topic, CompressedImage, self.img_callback, queue_size=1)

    @torch.no_grad()
    def img_callback(self, ros_data):
        #### direct conversion to CV2 ####
        # np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        # img_msg = ros_data.data
        img_msg = ros_data
        img = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Resize image to (112 x 112) and normalize
        # img = cv2.resize(image_np.copy(), dsize=(112, 112), interpolation=cv2.INTER_CUBIC).swapaxes(0, 2)
        img = cv2.resize(img.copy(), dsize=(112, 112), interpolation=cv2.INTER_CUBIC).swapaxes(0, 2)
        img = img / 255

        tensor = torch.as_tensor(img)
        ROS_INFO(f'input tensor dtype {tensor.dtype}')
        # ROS_INFO(f'weight dtype {self.model.segment.encoder.layers[0].weight.dtype}')
        tensor = tensor.unsqueeze(0).float()  # batch dim

        segm = self.model.segment.predict(tensor)
        segm_img = colorize_segmentation(segm.cpu().numpy())

        offset, phi = self.model.lanepos.predict(tensor)
        lanepose = LanePose(offset=offset.item(), phi=phi.item())

        _, pedest = self.model.pedest.predict(tensor)
        pedest = pedest.item()

        # Publish messages
        # self.segm_pub.publish(msg)

        # Create CompressedIamge for segmentation
        segm_img_msg = CompressedImage()
        segm_img_msg.header.stamp = rospy.Time.now()
        segm_img_msg.format = "jpeg"
        segm_img_msg.data = np.array(cv2.imencode('.jpg', segm_img)[1]).tostring()

        # Publish new image
        self.segm_vis_pub.publish(segm_img_msg)
        self.lanepose_pub.publish(lanepose)
        self.pedest_pub.publish(pedest)


if __name__ == '__main__':
    import pathlib
    cur_dir = pathlib.Path(__file__).absolute().parent
    duckie_name = 'yoneduckie'
    model_path = cur_dir / 'model.pt'
    print('model_path', model_path)
    detector = Detector(duckie_name, model_path=model_path)
    rospy.spin()