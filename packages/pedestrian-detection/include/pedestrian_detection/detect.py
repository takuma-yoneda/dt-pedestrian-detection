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

    ROS_INFO(f'segmentation input shape: {segm.shape}')  # (1, 7, 56, 56)
    segm = segm.squeeze()
    out = preset_colors[segm.argmax(axis=0)]
    ROS_INFO(f'segmentation output shape: {out.shape}')  # (56, 56, 3)
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
    def __init__(self, duckie_name, model_path) -> None:
        # assert torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available:
            ROS_INFO('CUDA is available!!')

        rospy.init_node('pedestrian_detection')

        # Load the pretrained model
        ROS_INFO(f'loading pretrained model from {model_path}...')
        self.model = load_model(model_path, self.device)
        ROS_INFO(f'loading pretrained model from {model_path}...done')

        # self.segm_pub = rospy.Publisher("/detection/segmentation", queue_size=1)
        # self.segm_vis_pub = rospy.Publisher("/detection/segmentation_vis", CompressedImage, queue_size=1)  # Just for visualization
        self.segm_vis_pub = rospy.Publisher("~segmentation_vis", Image, queue_size=1)  # Just for visualization
        self.lanepose_pub = rospy.Publisher("~lanepose", LanePose, queue_size=1)
        self.pedest_pub = rospy.Publisher("~pedestrian", Float32, queue_size=1)

        # DEBUG
        self.resized_img_pub = rospy.Publisher("~resized_img", Image, queue_size=1)

        # DEBUG
        # self.reconstr_vis_pub = rospy.Publisher("/detection/reconstr_img__vis", CompressedImage, queue_size=1)

        # TODO: fix it to handle general duckie names. Is there rosparam for it??
        image_topic = f'/{duckie_name}/camera_node/image/compressed'  # rospy.get_param("~joy_topic", '/joy_teleop/joy')
        self.img_sub = rospy.Subscriber(image_topic, CompressedImage, self.img_callback, queue_size=1)

    @torch.no_grad()
    def img_callback(self, compr_img_msg):
        #### direct conversion to CV2 ####
        # np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        # img_msg = ros_data.data
        _img = cvbridge.compressed_imgmsg_to_cv2(compr_img_msg, desired_encoding="bgr8")

        # Resize image to (112 x 112) and normalize
        img = cv2.resize(_img.copy(), dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(_img.copy(), dsize=(112, 112), interpolation=cv2.INTER_AREA)  # INTER_AREA is better than INTER_CUBIC
        # img = cv2.GaussianBlur(img, (5,5), 0)  # This deteriorates sgementation performance
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        img_resized = cvbridge.cv2_to_imgmsg(img, encoding="rgb8")
        self.resized_img_pub.publish(img_resized)

        # (height, width, channel) -> (channel, width, height)  (?)
        img = img / 255
        img = img.transpose(2, 0, 1)
        import time
        now = time.time()
        tensor = torch.as_tensor(img)
        ROS_INFO(f'input tensor dtype {tensor.dtype}')
        ROS_INFO('hello')
        # ROS_INFO(f'weight dtype {self.model.segment.encoder.layers[0].weight.dtype}')
        tensor = tensor.unsqueeze(0).float().to(self.device)  # Add batch dim
        ROS_INFO(f'input device {tensor.device}')

        segm = self.model.segment.predict(tensor)
        segm_img = colorize_segmentation(segm.cpu().numpy())

        offset, phi = self.model.lanepos.predict(tensor)
        lanepose = LanePose(offset=offset.item(), phi=phi.item())

        _, pedest = self.model.pedest.predict(tensor)
        ROS_INFO(f'pedest.item(): {pedest.item()}')
        # pedest = Bool(data=bool(pedest.item()))
        pedest = pedest.item()

        elapsed = time.time() - now
        ROS_INFO(f'elapsed: {elapsed}')

        # Publish messages
        # self.segm_pub.publish(msg)

        # Create CompressedIamge for segmentation
        # segm_img_msg = CompressedImage()
        # segm_img_msg.header.stamp = rospy.Time.now()
        # segm_img_msg.format = "jpeg"
        # segm_img_msg.data = np.array(cv2.imencode('.jpg', segm_img)[1]).tostring()
        # segm_img_msg = cvbridge.cv2_to_compressed_imgmsg(segm_img)

        # NOTE: Look at https://github.com/whats-in-a-name/CarND-Capstone/commit/de9ad68f4e5f1f983dd79254a71a51894946ac11
        segm_img_msg = cvbridge.cv2_to_imgmsg(segm_img, encoding="rgb8")

        # TEST
        # img = cvbridge.compressed_imgmsg_to_cv2(compr_img_msg, desired_encoding="bgr8")
        # ROS_INFO(f'original image shape: {_img.shape}')
        # ROS_INFO(f'original image dtype: {_img.dtype}')
        # reconstructed_img_msg = cvbridge.cv2_to_compressed_imgmsg(_img)
        # reconst_img_msg = CompressedImage()
        # reconst_img_msg.header.stamp = rospy.Time.now()
        # reconst_img_msg.format = "jpeg"
        # reconst_img_msg.data = np.array(cv2.imencode('.jpg', _img)[1]).tostring()

        # self.reconstr_vis_pub.publish(reconst_img_msg)

        # Publish new image
        self.segm_vis_pub.publish(segm_img_msg)
        self.lanepose_pub.publish(lanepose)
        self.pedest_pub.publish(pedest)

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
    # duckie_name = os.getenv('VEHICLE_NAME')
    duckie_name = 'yoneduckie'
    print('duckie_name', duckie_name)
    model_path = cur_dir / 'model.pt'
    print('model_path', model_path)
    detector = Detector('pedestrian_detection', duckie_name, model_path=model_path)
    # detector.test_pretrained_model('/test_imgs')
    rospy.spin()
