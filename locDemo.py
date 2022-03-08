"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch
import json
import random
import numpy as np
import cv2
from PIL import Image

from dataset.dataset_configs import STICKS
from experiment import init_model_from_dir
from tools.model_io import download_model
from tools.utils import get_net_input
from tools.vis_utils import matplot_plot_point_cloud 

from tools.vis_utils import show_projections
from visuals.rotating_shape_video import rotating_3d_video

combPts = []
combSticks = []

def stitchSave(images, savePath):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im.resize((im.size[0], max_height)), (x_offset,0))
        x_offset += im.size[0]

    new_im.save(savePath)

def run_demo(model, model_dir, data, idx):
    net_input = get_net_input(getTestSample(data, idx))

    preds = model(**net_input)
    canonical = preds['phi']['shape_canonical'][0]
    camera = preds['phi']['shape_camera_coord'][0]
    kp_loc_pred = preds['kp_reprojected_image'][0]
 
    # input keypoints
    kp_loc = net_input['kp_loc'][0]
    img = net_input['img']
    imgPath = net_input['imgPath']

    # predicted 3d keypoints in camera coords
    kp_pred_3d = preds['shape_image_coord'][0]

    sticks = STICKS['cars']

    # viz = get_visdom_connection()
    im_proj = show_projections(
        kp_loc[None].detach().cpu().numpy(),
        visdom_env='demo_h36m',
        visdom_win='input_keypoints',
        image_path=imgPath,
        title='input_keypoints',
        cmap__='rainbow',
        markersize=40,
        sticks=sticks,
        stickwidth=2,
    )

    im_proj_pred = show_projections(
        kp_loc_pred[None].detach().cpu().numpy(),
        visdom_env='demo_h36m',
        visdom_win='input_keypoints',
        image_path=imgPath,
        title='input_keypoints',
        cmap__='rainbow',
        markersize=40,
        sticks=sticks,
        stickwidth=2,
    )

    saveDir = os.path.join(model_dir, 'test%d' % idx)
    if not os.path.exists(saveDir):
       os.makedirs(saveDir)

    im_proj = Image.fromarray(im_proj)
    im_proj_pred = Image.fromarray(im_proj_pred)
    savePath = os.path.join(saveDir, 'projection.png')
    print('Saving keypoints to %s' % savePath)
    stitchSave([im_proj, im_proj_pred], savePath)

    video_path = os.path.join(saveDir, 'demo_shape.mp4')
    rotating_3d_video(camera.detach().cpu(),
                      video_path=video_path,
                      fps=1,
                      vlen=1,
                      sticks=sticks,
                      title='camera',
                      cmap='rainbow',
                      visdom_env=None,
                      visdom_win='3d_shape',
                      get_frames=20, )

    video_path = os.path.join(saveDir, 'demo_canonical_shape.mp4')
    rotating_3d_video(canonical.detach().cpu(),
                      video_path=video_path,
                      fps=1,
                      vlen=1,
                      sticks=sticks,
                      title='canonical',
                      cmap='rainbow',
                      visdom_env=None,
                      visdom_win='3d_shape',
                      get_frames=20, )

    # makeVid(saveDir)
    camPath = os.path.join(saveDir, 'demo_shape3D_0000.png')
    canonPath = os.path.join(saveDir, 'demo_canonical_shape3D_0000.png')
    inpImg = Image.open(imgPath)
    cam = Image.open(camPath)
    canon = Image.open(canonPath)
    savePath3D = os.path.join(saveDir, 'camera_vs_canonical.png')
    stitchSave([inpImg, cam, canon], savePath3D)

    os.system('cp ' + saveDir + '/projection.png finalRes/projection' + str(idx) + '.png')
    os.system('cp ' + saveDir + '/camera_vs_canonical.png finalRes/camera_vs_canonical' + str(idx) + '.png')
    os.system('rm -rf ' + saveDir)

    shape = canonical.detach().cpu()
    mean = shape.sum(1) / shape.shape[1]
    shape = shape - mean[:, None]
    global combPts, combSticks

    if len(combPts):
       combPts = np.concatenate((combPts, shape.numpy()), axis=1)
       combSticks = np.concatenate((combSticks, idx * np.shape(shape)[1] + np.array(sticks)))
    else:
       combPts = shape.numpy()
       combSticks = np.array(sticks)

def getTestSample(data, idx):
    kp_loc = data['data'][idx]['kp_loc']
    kp_vis = data['data'][idx]['kp_vis']
    imgPath = data['data'][idx]['img']
    img = Image.open(imgPath).convert('RGB')
    kp_loc, kp_vis = [torch.FloatTensor(a) for a in (kp_loc, kp_vis)]

    return {'kp_loc': kp_loc[None], 'kp_vis': kp_vis[None], 'imgPath': imgPath, 'img': img}

def makeVid(saveDir):
    imgArr = []

    for dirpath, dirs, files in os.walk(saveDir):
        for file in sorted(files):
            if "demo_shape3D" in file:
               img = cv2.imread(os.path.join(saveDir, file))
               imgArr.append(img)
               size = (img.shape[1], img.shape[0])

    out = cv2.VideoWriter(os.path.join(saveDir, 'camera.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
    for i in range(len(imgArr)):
        out.write(imgArr[i])

    out.release()

def main():
    f = open('./data/datasets/c3dpo/cars_test2.json')
    data = json.load(f)
    f.close()

    model_dir = download_model('cars')
    model, _ = init_model_from_dir(model_dir)
    model.eval()

    for i in range(0, 100):
        run_demo(model, model_dir, data, i)

    global combPts, combSticks
    fig = matplot_plot_point_cloud(combPts,
                                   pointsize=300, azim=-90, elev=90,
                                   figsize=(8, 8), title='combined_canonical',
                                   sticks=combSticks, lim=None,
                                   cmap='rainbow', ax=None, subsample=None,
                                   flip_y=True, savePath='finalRes/combined_canonical.png')

if __name__ == '__main__':
    main()
