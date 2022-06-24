#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pedestrian_detection.helper import ROS_INFO, cvbridge
import rospy

from std_msgs.msg import Float32
from pedestrian_detection.msg import LanePose

from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField, CompressedImage
from duckietown_msgs.msg import BoolStamped
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
    ], dtype=np.uint8)

    segm = segm.squeeze()  # (1, 7, 56, 56)
    out = preset_colors[segm.argmax(axis=0)]  # (56, 56, 3)
    return out


def load_model(path: str, device):
    """Load each state dict"""
    from pedestrian_detection.model import make_model
    joint_model = make_model().to(device)
    state_dicts = torch.load(path, map_location=device)
    key2module = {
        "encoder": joint_model.encoder,
        "pedest_lane": joint_model.pedest.lane_head,
        "lanepos_offset": joint_model.lanepos.offset_head,
        "lanepos_phi": joint_model.lanepos.phi_head,
        "segm": joint_model.segment.segm_head
    }
    for key, module in key2module.items():
        module.load_state_dict(state_dicts[key])

    return joint_model


class Detector:
    def __init__(self, node_name, model_path) -> None:
        """
        Subscriber:
            ~image (:obj:`sensor_msgs.msg.CompressedImage`): Input image
        Publishers:
            ~detection (:obj:`boolStamped`): Pedestrian Detection Flag
        """
        from collections import deque
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available:
            ROS_WARN('CUDA is NOT available!! The execution can be horribly slow')

        rospy.init_node(node_name)


        # Load the pretrained model
        ROS_INFO(f'loading pretrained model from {model_path}...')
        self.model = load_model(model_path, self.device)
        ROS_INFO(f'loading pretrained model from {model_path}...done')

        # Parameters
        self.detection_threshold = rospy.get_param("~detection_threshold", 0.7)
        self.time_window = rospy.get_param("~time_window", 3)

        ROS_INFO(f'detection threshold: {self.detection_threshold}')
        ROS_INFO(f'time window: {self.time_window}')

        self.last_stamp = rospy.Time.now()
        self.process_frequency = 10  # NOTE: not sure if this is reasonable...  # TODO: Use rosparam
        self.publish_duration = rospy.Duration.from_sec(1.0 / self.process_frequency)

        # self.segm_pub = rospy.Publisher("/detection/segmentation", queue_size=1)
        # self.segm_vis_pub = rospy.Publisher("/detection/segmentation_vis", CompressedImage, queue_size=1)  # Just for visualization
        # self.segm_vis_pub = rospy.Publisher("~segmentation_vis", Image, queue_size=1)  # Just for visualization
        # self.lanepose_pub = rospy.Publisher("~lanepose", LanePose, queue_size=1)
        self.img_sub = rospy.Subscriber("~img_compressed", CompressedImage, self.cb_image, queue_size=1, buff_size=20 * 1024 ** 2)
        self.pedest_pub = rospy.Publisher("~detection_score", Float32, queue_size=1)
        self.pub_detection_flag = rospy.Publisher("~detection", BoolStamped, queue_size=1)

        self._filtering = True
        self._past_detections = deque([False] * self.time_window, maxlen=self.time_window)


    @staticmethod
    def process_image(image_cv, dsize, interpolation=cv2.INTER_CUBIC):
        """
        Resize, normalize and correct axes of the input cv2 image, so that torch can handle it easily.
        """
        image_cv = cv2.resize(image_cv.copy(), dsize=dsize, interpolation=interpolation)
        image_cv = cv2.cvtColor(image_cv.copy(), cv2.COLOR_BGR2RGB)

        # (width, height, channel) -> (channel, width, height)
        return image_cv.transpose(2, 0, 1) / 255

    @torch.no_grad()
    def cb_image(self, image_msg):
        """
        Callback for processing a image which potentially contains a back pattern. Processes the image only if
        sufficient time has passed since processing the previous image (relative to the chosen processing frequency).
        The pattern detection is performed using OpenCV's `findCirclesGrid <https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=solvepnp#findcirclesgrid>`_ function.
        Args:
            image_msg (:obj:`sensor_msgs.msg.CompressedImage`): Input image
        """
        now = rospy.Time.now()
        if now - self.last_stamp < self.publish_duration:
            return
        else:
            self.last_stamp = now

        image_cv = cvbridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")

        # TODO: Create a rosparam for dsize?
        img = self.process_image(image_cv, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)

        tensor = torch.as_tensor(img)
        tensor = tensor.unsqueeze(0).float().to(self.device)  # Add batch dim

        # Predict segmentation
        # segm = self.model.segment.predict(tensor)
        # segm_img = colorize_segmentation(segm.cpu().numpy())

        # Predict lane offset & angle
        # offset, phi = self.model.lanepos.predict(tensor)
        # lanepose = LanePose(offset=offset.item(), phi=phi.item())

        # Detect pedestrian
        _, pedest_prob = self.model.pedest.predict(tensor)
        ROS_INFO(f'pedest_prob: {pedest_prob.item()}')
        pedest_prob = pedest_prob.item()

        detection = pedest_prob > self.detection_threshold
        self._past_detections.appendleft(detection)

        detection_flag_msg = BoolStamped()
        detection_flag_msg.header = image_msg.header
        if not self._filtering:
            detection_flag_msg.data = detection
        else:
            detection_flag_msg.data = all(self._past_detections)

        self.pub_detection_flag.publish(detection_flag_msg)

        # Publish new image
        # NOTE: Look at https://github.com/whats-in-a-name/CarND-Capstone/commit/de9ad68f4e5f1f983dd79254a71a51894946ac11
        # segm_img_msg = cvbridge.cv2_to_imgmsg(segm_img, encoding="rgb8")
        # self.segm_vis_pub.publish(segm_img_msg)
        # self.lanepose_pub.publish(lanepose)
        self.pedest_pub.publish(pedest_prob)

    @torch.no_grad()
    def test_pretrained_model(self, data_dir):
        """Test the pretrained model with an offline data."""
        from os.path import join as pjoin
        from pathlib import Path
        ROS_INFO('=== Running test_pretrained_model ===')
        ROS_INFO(f'data_dir: {data_dir}, {len(list(Path(data_dir).iterdir()))}')
        target_dir = Path(data_dir) / 'test'
        target_dir.mkdir(exist_ok=True)
        for fpath in Path(data_dir).iterdir():
            if fpath.suffix not in ['.png', '.jpg']:
                continue
            ROS_INFO(str(fpath))
            image_cv = cv2.imread(str(fpath))

            img = self.process_image(image_cv, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)

            tensor = torch.as_tensor(img)
            tensor = tensor.unsqueeze(0).float().to(self.device)  # Add batch dim

            # Perform inference
            segm = self.model.segment.predict(tensor)
            segm_img = colorize_segmentation(segm.cpu().numpy())

            offset, phi = self.model.lanepos.predict(tensor)
            lanepose = LanePose(offset=offset.item(), phi=phi.item())

            _, pedest_prob = self.model.pedest.predict(tensor)
            ROS_INFO(f'pedest_prob: {pedest_prob.item()}')

            # Annotate the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 255)
            # NOTE: Somehow “copy” is critical (https://stackoverflow.com/q/23830618/7057866)
            annotated_img = cv2.resize(image_cv, dsize=(224, 224))
            annotated_img = cv2.putText(annotated_img, f'duck-on-lane: {pedest_prob.item():.2f}', (0, 20), font, fontScale, color)
            annotated_img = cv2.putText(annotated_img, f'offset: {offset.item():.2f}', (0, 40), font, fontScale, color)
            annotated_img = cv2.putText(annotated_img, f'phi: {phi.item():.2f}', (0, 60), font, fontScale, color)

            cv2.imwrite(str(target_dir / fpath.name), annotated_img)
        ROS_INFO('test completed!')



if __name__ == '__main__':
    import pathlib
    import os
    cur_dir = pathlib.Path(__file__).absolute().parent
    model_path = cur_dir / 'model.pt'
    ROS_INFO(f'model_path: {model_path}')
    detector = Detector('pedestrian_detection', model_path=model_path)
    # detector.test_pretrained_model('/test_imgs')
    rospy.spin()
