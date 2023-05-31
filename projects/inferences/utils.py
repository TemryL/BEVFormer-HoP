import cv2
import torch
import numpy as np
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


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
    """Make inference on a single image.
    Args:
        model (nn.Module): The loaded model.
        model_cfg (Config): The model config.
        img_cfg (Config): The image config.
        img (ndarray): The image to be inferred. If None, will load from img_cfg.path.
    """
    # Load and prepare image
    if img is None:
        img = cv2.imread(img_cfg.path)
    img = imnormalize_(np.float32(img), **img_cfg.norm_cfg)
    img_shape = img.shape
    img = torch.tensor(img)
    img = img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    
    # Simulate 6 surrounding images wih 5 random img + 1 front image
    rand = torch.empty(img.size()).normal_(mean=0, std=img.std())
    front = img
    front_right = rand
    front_left = rand
    back_right = rand
    back_left = rand
    back = rand
    imgs = torch.cat([back, back_right, back_left, front, front_left, front_right], dim=1)

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
    """Render the inference result on a BEV image.
    Args:
        pts_bbox (dict): The inference result.
        model_cfg (Config): The model config.
        score_tresh (float): The score threshold to render.
    """
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