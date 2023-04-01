import os
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import torch

from data import cfg_mnet, cfg_re50
from network import RetinaFace_HPE
from test import load_model, eval_single, get_det_landm_image, get_hpe_image
from utils import utils


def parser():
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='./weights/BIWI,Pose_300W_LP,BIWI_masked,Pose_300W_LP_maskedResnet50_epoch_80.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--img', help='image path', default='./samples/frame_00357_rgb.png')
    parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')

    # parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    # parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    # parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
    # parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    # parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg = None

    args = parser()

    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    net = RetinaFace_HPE(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model)
    net.eval()
    print('Finished loading model!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    img_path = args.img
    img_raw = io.imread(img_path)

    rgb_mean_normalize = [123, 117, 104]
    img = img_raw - rgb_mean_normalize
    img = img / 255.
    img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    dets, landms, poses = eval_single(net, img, cfg)

    pose = torch.tensor(poses[0][:1])
    pred_poses = utils.compute_euler_angles_from_rotation_matrices(pose, use_gpu=False) * 180 / np.pi

    img_det_ldmk = img_raw.copy()
    img_hpe = img_raw.copy()
    img_det_ldmk = get_det_landm_image(img_det_ldmk, dets[0], landms[0], vis_thres=cfg['vis_thres'])
    img_hpe = get_hpe_image(img_hpe, dets[0], pred_poses[0].reshape(1, 3).numpy(), vis_thres=cfg['vis_thres'])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_det_ldmk); ax[0].set_title('face detection'); ax[0].axis('off')
    ax[1].imshow(img_hpe); ax[1].set_title('head pose estimation'); ax[1].axis('off')
    plt.show()