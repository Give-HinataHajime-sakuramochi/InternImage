from argparse import ArgumentParser
#在 InternImage 環境下 run
import mmcv

import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time



# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def Point_T(ori_x, ori_y, top_x, top_y):
    radian1 = math.atan2(top_y - ori_y, top_x - ori_x)
    angle1 = (radian1 * 180) / math.pi
    return angle1



def pipline(model, img):
    segment_image = inference_segmentor(model, img)

    # 把非0的全部同一為1
    segment_image = np.array(segment_image[0]).astype(np.uint8)
    segment_image[segment_image == 1] = 0  # 把sideroad和road 合成一個類別
    gray_img = np.minimum(segment_image, 1) * 255

    #Canny邊偵測
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    res = np.where(edges == 255)
    canny_edge= list(zip(res[1], res[0]))
    canny_edge=np.array(canny_edge)
    
    # 搜尋輪廓法
    #contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## 輪廓平滑化
    approx=cv2.approxPolyDP(canny_edge, 15, False);
    edges = np.zeros_like(segment_image)
    #cv2.drawContours(edges, [approx], 0, 255, 2, 8);  #

    #找出縱軸最高點
    #res = np.where(edges == 255)  #1
    #coordinates= list(zip(res[1], res[0]))  #1
    origin = (int(edges.shape[1]/2), edges.shape[0])
    top_point = (edges.shape[1],edges.shape[0])
    for coordinate in approx: #1in coordinates
        coordinate = coordinate.flatten()
        cv2.circle(edges, (coordinate[0], coordinate[1]), 3, (255, 255, 255), 4)
        if coordinate[1] < top_point[1] and coordinate[1]>10 and coordinate[0]>10 and coordinate[0]<(edges.shape[1]-10):####
            top_point = coordinate
    cv2.line(edges, origin, top_point, 255, 2)
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    edges_img = weighted_img(edges,img,α=0.8, β=1., γ=0.)
    ori_x,ori_y=img.shape[1]/2,img.shape[0]
    angle=Point_T(ori_x, ori_y, top_point[0], top_point[1])
    cv2.putText(edges_img, str(angle), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    if abs(angle+90)<15:
        cv2.putText(edges_img, "Straight", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    elif abs(angle)<90:
        cv2.putText(edges_img, "Turn left", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    else:
        cv2.putText(edges_img, "Turn right", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

    return edges_img


def main():
    print("start")
    #改相關檔案路徑
    parser = ArgumentParser()
    parser.add_argument('--img', default="image/image.jpg",
                        help='Image file')
    parser.add_argument('--config',
                        default="configs/cityscapes/upernet_internimage_xl_512x1024_160k_mapillary2cityscapes.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default="checkpoint/cityscapes.pth",
                        help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    # test a single image
    '''
    img = cv2.imread(args.img)
    edge_img= pipline(model, img)
    plt.figure(figsize=(10, 5))
    plt.imshow(edge_img)
    plt.savefig("output1.jpg")
    '''
    # test 
    cap = cv2.VideoCapture("image/30427_hd_Trim_Trim.mp4")
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('edge_direction_demo.avi', fourcc, 20.0, (video_w, video_h))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==False:
            break
        test_images_output =  pipline(model,frame)
        out.write(test_images_output)
        #print(i)
        i+=1

    cap.release()
    out.release()
    print("OK")

if __name__ == '__main__':
    main()
