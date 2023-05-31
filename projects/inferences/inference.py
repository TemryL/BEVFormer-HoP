import cv2
import argparse

from tqdm import tqdm
from mmcv import Config
from ..mmdet3d_plugin import *
from mmdet.apis.inference import init_detector
from .utils import make_inference, render_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make inference on video or single image using trained model.'
    )
    parser.add_argument('model_cfg_file', help='path to model config file')
    parser.add_argument('checkpoint_file', help='path to checkpoint file')
    parser.add_argument('img_cfg_file', help='path to image config file')
    parser.add_argument('score_tresh', help='score threshold for rendering', type=float)
    return parser.parse_args()


def main(model_cfg_file, checkpoint_file, img_cfg_file, score_tresh):
    """Main function for inference on video or single image
    
    Args:
        model_cfg_file (str): path to model config file
        checkpoint_file (str): path to checkpoint file
        img_cfg_file (str): path to image config file
        score_tresh (float): score threshold for rendering
    """
    # Load configs files
    model_cfg = Config.fromfile(model_cfg_file)
    img_cfg = Config.fromfile(img_cfg_file)
    
    # Init model
    model = init_detector(model_cfg, checkpoint_file)
    
    bev_maps = []
    pts_bboxs = []
    img_bevs = []
    # Make inference on video or single image
    if img_cfg.path.endswith('.mp4'):
        cap = cv2.VideoCapture(img_cfg.path)
        nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        result = cv2.VideoWriter(img_cfg.out_dir, 
                                cv2.VideoWriter_fourcc(*'MP4V'),
                                fps, (model_cfg.bev_w_, model_cfg.bev_h_))
        
        with tqdm(total=nb_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                else:
                    bev_feature_map, pts_bbox = make_inference(model, model_cfg, img_cfg, img=frame)
                    img_bev = render_inference(pts_bbox, model_cfg, score_tresh=score_tresh)
                    result.write(img_bev)
                    bev_maps.append(bev_feature_map)
                    pts_bboxs.append(pts_bbox)
                    img_bevs.append(img_bev)
                
                pbar.update(1)
            
        cap.release()
    else:
        bev_feature_map, pts_bbox = make_inference(model, model_cfg, img_cfg)
        img_bev = render_inference(pts_bbox, model_cfg, score_tresh=score_tresh)
        cv2.imwrite(img_cfg.out_dir, img_bev)
        bev_maps.append(bev_feature_map)
        pts_bboxs.append(pts_bbox)
        img_bevs.append(img_bev)
    
    return bev_maps, pts_bboxs, img_bevs


if __name__ == '__main__':
    args = parse_args()
    bev_maps, pts_bboxs, img_bevs = main(args.model_cfg_file, args.checkpoint_file, args.img_cfg_file, args.score_tresh)