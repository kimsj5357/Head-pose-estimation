from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from network import RetinaFace_HPE
from datasets import getDataset
from utils import utils
from tqdm import tqdm
from skimage import io
from PIL import Image

def parser():
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model',
                        # default='./weights/BIWI,Pose_300W_LP__Resnet50_Final.pth',
                        # default='./weights/BIWI,Pose_Para_300W_LP,BIWI_masked,Pose_300W_LP_masked__Resnet50_Final.pth',
                        default='./weights/BIWI,BIWI_rotated__Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        choices=['BIWI', 'BIWI_masked', 'AFLW2000', 'AFLW2000_masked', 'Pose_300W_LP', 'Pose_300W_LP_masked'],
        default='BIWI_masked', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--save_folder', default='./results', type=str, help='Dir to save results')
    parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
    args = parser.parse_args()

    return args


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu=False):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def eval_single(model, img, cfg):
    device = img.device
    b, _, im_height, im_width = img.shape

    with torch.no_grad():
        loc, conf, landms, poses = model(img)  # forward pass

    # bbox, landmarks
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    prior_data = prior_data.repeat(b, 1, 1)

    boxes = decode(loc.data, prior_data, cfg['variance'])
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    boxes = boxes * scale.to(device)
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, :, 1]
    landms = decode_landm(landms.data, prior_data, cfg['variance'])
    scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()
    poses = poses.cpu().numpy()

    # ignore low scores

    pred_boxes, pred_landms, pred_poses = [], [], []

    for i in range(b):
        inds = np.where(scores[i] > cfg['confidence_threshold'])[0]
        box = boxes[i, inds]
        landm = landms[i, inds]
        score = scores[i, inds]
        pose = poses[i, inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:args.top_k]
        order = score.argsort()[::-1]
        box = box[order]
        landm = landm[order]
        score = score[order]
        pose = pose[order]

        # do NMS
        dets = np.hstack((box, score[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, cfg['nms_threshold'])

        dets = dets[keep]
        landm = landm[keep]
        pose = pose[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        pred_boxes.append(dets)
        pred_landms.append(landm)
        pred_poses.append(pose)

    return pred_boxes, pred_landms, pred_poses


def eval(model, dataloader, cfg, save_path=None):

    total = 0
    yaw_error = pitch_error = roll_error = 0.

    for i, (img, target, gt_pose, pose_label, img_name) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img_raw = img.clone()
        img = img.float().cuda()

        rgb_mean_normalize = torch.tensor([123, 117, 104]) / 255.
        img = img - rgb_mean_normalize.cuda()
        img = img.permute(0, 3, 1, 2)

        dets, landms, poses = eval_single(model, img, cfg)

        gt_poses = pose_label * 180 / np.pi
        pred_poses = torch.zeros_like(gt_poses)
        for i in range(img.size(0)):
            if len(poses[i]) > 0:
                pose = torch.tensor(poses[i][:1])
                euler = utils.compute_euler_angles_from_rotation_matrices(pose) * 180 / np.pi
                pred_poses[i] = euler

        p_error, y_error, r_error = eval_pose(pred_poses, gt_poses)

        total += img.size(0)
        pitch_error += p_error.item()
        yaw_error += y_error.item()
        roll_error += r_error.item()

        if save_path is not None:
            img_raw = np.array(img_raw * 255).astype(np.uint8)

            # n_images = 4
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(3, n_images)

            for i in range(img.size(0)):
                img_det_ldmk = img_raw[i].copy()
                img_hpe = img_raw[i].copy()
                img_gt_hpe = img_raw[i].copy()

                img_det_ldmk = get_det_landm_image(img_det_ldmk, dets[i], landms[i], vis_thres=cfg['vis_thres'])
                img_hpe = get_hpe_image(img_hpe, dets[i], pred_poses[i].reshape(1, 3).numpy(), vis_thres=cfg['vis_thres'])

                img_gt_hpe = get_hpe_image(img_gt_hpe, dets[i], gt_poses[i].reshape(1, 3).numpy(), vis_thres=cfg['vis_thres'])
                # img_gt_ldmk = get_det_landm_image(img_raw[i].copy(),
                #                                   target[i][:, :5].numpy(),
                #                                   target[i][:, 4:-1].numpy(),
                #                                   vis_thres=cfg['vis_thres'])

            #     if i >= n_images:
            #         continue
            #     ax[0][i].imshow(img_det_ldmk); ax[0][i].axis('off')
            #     ax[1][i].imshow(img_hpe); ax[1][i].axis('off')
            #     ax[2][i].imshow(img_gt_hpe); ax[2][i].axis('off')
            # plt.tight_layout()
            # plt.show()

                save_name = "./results/BIWI_masked/" + '/'.join(img_name[i].split(',')[0].split('/')[-2:])
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                # io.imsave(save_name, img_hpe)
                save_img = Image.fromarray(img_hpe)
                save_img.save(save_name)

    error_log = 'Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
        yaw_error / total, pitch_error / total, roll_error / total,
        (yaw_error + pitch_error + roll_error) / (total * 3))
    # print('Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (
    #     yaw_error / total, pitch_error / total, roll_error / total,
    #     (yaw_error + pitch_error + roll_error) / (total * 3)))
    return error_log


def eval_pose(pred_pose_label, gt_pose_label):
    # pose shape: (b, 3)

    y_gt_deg = gt_pose_label[:, 0]
    p_gt_deg = gt_pose_label[:, 1]
    r_gt_deg = gt_pose_label[:, 2]

    y_pred_deg = pred_pose_label[:, 0]
    p_pred_deg = pred_pose_label[:, 1]
    r_pred_deg = pred_pose_label[:, 2]

    pitch_error = torch.sum(torch.min(
        torch.stack((torch.abs(p_gt_deg - p_pred_deg), torch.abs(p_pred_deg + 360 - p_gt_deg), torch.abs(
            p_pred_deg - 360 - p_gt_deg), torch.abs(p_pred_deg + 180 - p_gt_deg),
                     torch.abs(p_pred_deg - 180 - p_gt_deg))), 0)[0])
    yaw_error = torch.sum(torch.min(
        torch.stack((torch.abs(y_gt_deg - y_pred_deg), torch.abs(y_pred_deg + 360 - y_gt_deg), torch.abs(
            y_pred_deg - 360 - y_gt_deg), torch.abs(y_pred_deg + 180 - y_gt_deg),
                     torch.abs(y_pred_deg - 180 - y_gt_deg))), 0)[0])
    roll_error = torch.sum(torch.min(
        torch.stack((torch.abs(r_gt_deg - r_pred_deg), torch.abs(r_pred_deg + 360 - r_gt_deg), torch.abs(
            r_pred_deg - 360 - r_gt_deg), torch.abs(r_pred_deg + 180 - r_gt_deg),
                     torch.abs(r_pred_deg - 180 - r_gt_deg))), 0)[0])

    return pitch_error, yaw_error, roll_error

def get_det_landm_image(img, bboxes, landms, vis_thres=0.5):
    """
    :param img: [h, w, c]
    :param bboxes: [num, 5(bbox + score)]
    :param landms: [num, 10]
    :return:
    """
    w, h, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bbox, landm in zip(bboxes, landms):
        if bbox[4] < vis_thres:
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            continue
        text = "{:.4f}".format(bbox[4])
        bbox = list(map(int, bbox))
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cx = bbox[0]
        cy = bbox[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        landm = list(map(int, landm))
        cv2.circle(img, (landm[0], landm[1]), 5, (0, 0, 255), 4)    # left eye (red)
        cv2.circle(img, (landm[2], landm[3]), 5, (0, 255, 255), 4)  # right eye (yellow)
        cv2.circle(img, (landm[4], landm[5]), 5, (255, 0, 255), 4)  # nose (pink)
        cv2.circle(img, (landm[6], landm[7]), 5, (0, 255, 0), 4)    # mouth left (green)
        cv2.circle(img, (landm[8], landm[9]), 5, (255, 0, 0), 4)    # mouth right (blue)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_hpe_image(img, bboxes, poses, vis_thres=0.5):
    """
    :param img: [h, w, c]
    :param bboxes: [num, 5(bbox + score)]
    :param poses: [num, 3]
    :return:
    """
    w, h, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for bbox, pose in zip(bboxes, poses):
        if bbox[4] < vis_thres:
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            continue
        bbox = list(map(int, bbox))

        y_pred_deg = pose[0]
        p_pred_deg = pose[1]
        r_pred_deg = pose[2]

        center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        utils.plot_pose_cube(img, y_pred_deg, p_pred_deg, r_pred_deg,
                             center_x, center_y, size=bbox[2] - bbox[0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None

    args = parser()

    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    cfg['confidence_threshold'] = args.confidence_threshold
    cfg['nms_threshold'] = args.nms_threshold

    # net and model
    net = RetinaFace_HPE(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    save_path = None
    if args.save_image:
        save_path = os.path.join(args.save_folder, args.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    test_dataset = getDataset(args.dataset, train_mode=False)
    num_images = len(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=8)

    net.eval()
    error_log = eval(net, test_dataloader, cfg, save_path=save_path)
    print(error_log)

