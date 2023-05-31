import cv2
import torch
import numpy as np

from tqdm import tqdm
from mmcv import Config, DictAction
from projects.mmdet3d_plugin import *
from mmdet.apis.inference import init_detector
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

import matplotlib.colors as mcolors

# BEV queries is
# 200×200, the perception ranges are [−51.2m, 51.2m] for the X and Y axis and the size of resolution
# s of BEV’s grid is 0.512m

def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def make_inference(model, model_cfg, img_cfg, img=None):
    # Load and prepare image
    if img is None:
        img = cv2.imread(img_cfg.path)
    img = imnormalize_(np.float32(img), **img_cfg.norm_cfg)
    img_shape = img.shape
    img = torch.tensor(img)
    img = img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    
    # Simulate 6 surrounding images wih 5 random img + 1 front image
    rand = 0*torch.empty(img.size()).normal_(mean=0, std=img.std())
    front = img
    front_right = img
    front_left = img
    back_right = img
    back_left = img
    back = img
    imgs = torch.cat([back, back_right, back_left, front, front_left, front_right], dim=1)
    #print(imgs.element_size() * imgs.nelement(), 'GBbbbbbbb')
    # Prepare meta data
    img_metas = dict()
    img_metas['can_bus'] = img_cfg.CAN_BUS
    img_metas['lidar2img'] = img_cfg.LIDAR2IMG
    img_metas['img_shape'] = [img_shape for i in range(6)]
    img_metas['box_type_3d'] = LiDARInstance3DBoxes
    
    # Make inference
    outs = model.simple_test(img_metas=[img_metas], img=imgs.to('cuda'))
    bev_feature_map = outs[0].permute(1, 0, 2).reshape(model_cfg.bev_h_, model_cfg.bev_w_, model_cfg._dim_)
    pts_bbox = outs[1][0]['pts_bbox']
    
    return bev_feature_map, pts_bbox

def render_inference(pts_bbox, model_cfg, score_tresh=0.15):
    colors = {
        'Purple': (128, 0, 128),
        'Green': (0, 255, 0),
        'Blue': (255, 0, 0),
        'Yellow': (0, 255, 255),
        'Magenta': (255, 0, 255),
        'Cyan': (255, 255, 0),
        'Maroon': (0, 0, 128),
        'Lime': (0, 128, 0),
        'Navy': (128, 0, 0),
        'Gray': (128, 128, 128),
        'Red': (0, 0, 255)
    }
    color_idx = {
        0: 'Purple',
        1: 'Green',
        2: 'Blue',
        3: 'Yellow',
        4: 'Magenta',
        5: 'Cyan',
        6: 'Maroon',
        7: 'Lime',
        8: 'Navy',
        9: 'Gray',
        10: 'Red'
    }
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    scores_3d = pts_bbox['scores_3d']
    labels_3d = pts_bbox['labels_3d']
    boxes_3d = pts_bbox['boxes_3d']
    bev_boxes = boxes_3d.bev
    
    H, W = model_cfg.bev_h_, model_cfg.bev_w_
    ego_pose = (int(H/2), int(W/2))
    bev_grid_size = 0.512
    img = 255*np.ones((model_cfg.bev_h_, model_cfg.bev_w_, 3), np.uint8)
    img = cv2.circle(img, ego_pose, 5, colors[color_idx[10]], -1)
    
    for i in range(bev_boxes.shape[0]):
        if (scores_3d[i] > score_tresh):
            label = labels_3d[i].item()
            bbox = bev_boxes[i,:].detach().numpy()
            X = bbox[0]/bev_grid_size
            Y = bbox[1]/bev_grid_size
            w = bbox[2]/bev_grid_size
            h = bbox[3]/bev_grid_size
            R = bbox[4]
            box_points = cv2.boxPoints(((X+W/2, Y+H/2), (w, h), R))
            box_points = np.int0(box_points)
            img = cv2.drawContours(img, [box_points], 0, colors[color_idx[label]], 2)

    return img


if __name__ == '__main__':
    
    model_cfg_file = 'projects/configs/bevformer/bevformer_base.py'
    img_cfg_file = 'video_configs/test1_configs.py'
    checkpoint_file = 'ckpts/bevformer_r101_dcn_24ep.pth'
    score_tresh = 0.15
    
    # Load configs files
    model_cfg = Config.fromfile(model_cfg_file)
    img_cfg = Config.fromfile(img_cfg_file)
    
    # Init model
    model = init_detector(model_cfg, checkpoint_file)

    # Make inference on video or single image
    if img_cfg.path.endswith('.mp4'):
        cap = cv2.VideoCapture(img_cfg.path)
        nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(img_cfg.out_dir, 
                                cv2.VideoWriter_fourcc(*'MP4V'),
                                fps, (model_cfg.bev_w_, model_cfg.bev_h_))
        # mean_r = []
        # mean_g = []
        # mean_b = []
        with tqdm(total=nb_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                else:
                    # mean_b.append(frame[:,:,0])
                    # mean_g.append(frame[:,:,1])
                    # mean_r.append(frame[:,:,2])
                    bev_feature_map, pts_bbox = make_inference(model, model_cfg, img_cfg, img=frame)
                    img_bev = render_inference(pts_bbox, model_cfg, score_tresh=score_tresh)
                    result.write(img_bev)
                
                pbar.update(1)
            
        cap.release()
    else:
        bev_feature_map, pts_bbox = make_inference(model, model_cfg, img_cfg)
        img_bev = render_inference(pts_bbox, model_cfg, score_tresh=score_tresh)
        cv2.imwrite(img_cfg.out_dir, img_bev)

    # B = np.array(mean_b).std()
    # G = np.array(mean_g).std()
    # R = np.array(mean_b).std()
    # print(B,G,R)
    