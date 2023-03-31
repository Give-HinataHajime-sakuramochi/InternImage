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

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    #left
    left_x = []
    left_y = []
    left_slope = []
    left_intercept = []

    #right
    right_x = []
    right_y = []
    right_slope = []
    right_intercept = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = cal_slope(x1,y1,x2,y2)
            if slope is not None and 0 < slope < 2:
                left_slope.append(cal_slope(x1,y1,x2,y2))
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
                left_intercept.append(y1 - x1*cal_slope(x1,y1,x2,y2))
            if slope is not None and -2 < slope < 0:
                right_slope.append(cal_slope(x1,y1,x2,y2))
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
                right_intercept.append(y1 - x1*cal_slope(x1,y1,x2,y2))
            #else continue
    # Line: y = ax + b
    # Calculate a & b by the two given line(right & left)
    average_left_slope = 0
    average_right_slope = 0
    left_y_min,left_x_min,right_y_min,right_x_min=0,0,0,0 # for calculate intersection
    #left
    if(len(left_x) != 0 and len(left_y)!= 0 and len(left_slope) != 0 and len(left_intercept)!= 0 ): 
        average_left_x = sum(left_x)/len(left_x)
        average_left_y = sum(left_y)/len(left_y)
        average_left_slope = sum(left_slope)/len(left_slope)
        average_left_intercept = sum(left_intercept)/len(left_intercept)   
        left_y_min = img.shape[0]*0.6
        left_x_min = (left_y_min - average_left_intercept)/average_left_slope
        left_y_max = img.shape[0]
        left_x_max = (left_y_max - average_left_intercept)/average_left_slope
        cv2.line(img, (int(left_x_min), int(left_y_min)), (int(left_x_max), int(left_y_max)), color, thickness)

    #right   
    if(len(right_x) != 0 and len(right_y)!= 0 and len(right_slope) != 0 and len(right_intercept)!= 0):
        average_right_x = sum(right_x)/len(right_x)
        average_right_y = sum(right_y)/len(right_y)
        average_right_slope = sum(right_slope)/len(right_slope)
        average_right_intercept = sum(right_intercept)/len(right_intercept)
        right_y_min = img.shape[0]*0.6
        right_x_min = (right_y_min - average_right_intercept)/average_right_slope
        right_y_max = img.shape[0]
        right_x_max = (right_y_max - average_right_intercept)/average_right_slope 
        cv2.line(img, (int(right_x_min), int(right_y_min)), (int(right_x_max), int(right_y_max)), color, thickness)
    intersection_x,intersection_y=0,0
    # cal intersection
    if average_left_slope!=0 and average_right_slope!=0:
        a1,b1,a2,b2=average_left_slope,-1,average_right_slope,-1
        c1=left_y_min-average_left_slope*left_x_min
        c2=right_y_min-average_right_slope*right_x_min
        intersection_x=(b1*c2-b2*c1)/(b2*a1-b1*a2)
        intersection_y =(a1*c2-c1*a2)/(b1*a2-a1*b2)
    return intersection_x,intersection_y


def cal_slope(x1, y1, x2, y2):
    if x2 == x1:  # devide by zero
        return None
    else:
        return ((y2 - y1) / (x2 - x1))


def intercept(x, y, slope):
    return y - x * slope


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is None:
        lines = []
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    intersection_x,intersection_y=draw_lines(line_img, lines)
    return line_img,intersection_x,intersection_y


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def Point_T(point_x, point_y, inter_x, inter_y):
    radian1 = math.atan2(inter_y - point_y, inter_x - point_x)
    angle1 = (radian1 * 180) / math.pi

    radian2 = math.atan2(point_y / 2 - point_y, point_x - point_x)
    angle2 = (radian2 * 180) / math.pi

    included_angle = abs(angle1 - angle2)
    turn_dir=""
    if abs(inter_x - point_x) < 10:
        turn_dir="Straight"
    elif inter_x - point_x > 0:
        turn_dir="Turn right"
    else:
        turn_dir="Turn left"
    return turn_dir,included_angle

def find_point(approx):
    tail=len(approx)
    left1_x,left1_y=approx[0][0][0],approx[0][0][1]
    left2_x,left2_y=approx[1][0][0],approx[1][0][1]
    right1_x,right1_y=approx[tail-1][0][0],approx[tail-1][0][1]
    right2_x,right2_y=approx[tail-2][0][0],approx[tail-2][0][1]
    left_slope,right_slope=(left2_y-left1_y)/(left2_x-left1_x),(right2_y-right1_y)/(right2_x-right1_x)
    for i in range(tail):
        if i==tail-1:
            tmp_s=(approx[i][0][1]-approx[0][0][1])/(approx[i][0][0]-approx[0][0][0])
            if tmp_s!=float('inf') and tmp_s!=0:
                if tmp_s<0:
                    #right，取最小
                    if tmp_s<right_slope:
                        right_slope=tmp_s
                        right1_x,right1_y=approx[i][0][0],approx[i][0][1]
                        right2_x,right2_y=approx[0][0][0],approx[0][0][1]
                elif tmp_s>0:
                    #left，取最大
                    if tmp_s>left_slope:
                        left_slope=tmp_s
                        left1_x,left1_y=approx[i][0][0],approx[i][0][1]
                        left2_x,left2_y=approx[0][0][0],approx[0][0][1]
                        
        else:
            tmp_s=(approx[i][0][1]-approx[i+1][0][1])/(approx[i][0][0]-approx[i+1][0][0])
            if tmp_s!=float('inf') and tmp_s!=0:
                if tmp_s<0:
                    #right，取最小
                    if tmp_s<right_slope or right_slope==float('inf'):
                        right_slope=tmp_s
                        right1_x,right1_y=approx[i][0][0],approx[i][0][1]
                        right2_x,right2_y=approx[i+1][0][0],approx[i+1][0][1]
                elif tmp_s>0:
                    #left，取最大
                    if tmp_s>left_slope or left_slope==float('inf'):
                        left_slope=tmp_s
                        left1_x,left1_y=approx[i][0][0],approx[i][0][1]
                        left2_x,left2_y=approx[i+1][0][0],approx[i+1][0][1]
    intersection_x,intersection_y=0,0
    if left_slope!=0 and right_slope!=0:
        a1,b1,a2,b2=left_slope,-1,right_slope,-1
        c1 = left1_y - left_slope * left1_x
        c2 = right1_y - right_slope * right1_x
        intersection_x=(b1*c2-b2*c1)/(b2*a1-b1*a2)
        intersection_y =(a1*c2-c1*a2)/(b1*a2-a1*b2)
    return intersection_x,intersection_y,left1_x,left1_y,left2_x,left2_y,right1_x,right1_y,right2_x,right2_y

def pipline(model, img):
    segment_image = inference_segmentor(model, img)

    # 把非0的全部同一為1
    segment_image = np.array(segment_image[0]).astype(np.uint8)
    segment_image[segment_image == 1] = 0  # 把sideroad和road 合成一個類別
    gray_img = np.minimum(segment_image, 1) * 255

    # #Canny邊偵測
    # low_threshold = 100
    # high_threshold = 150
    # edges = cv2.Canny(gray_img, low_threshold, high_threshold)

    # 搜尋輪廓法
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## 輪廓平滑化
    approx=cv2.approxPolyDP(contours[0], 15, False);
    edges = np.zeros_like(segment_image)
    cv2.drawContours(edges, [approx], 0, 255, 2, 8);

    #找出縱軸最高點
    '''
    res = np.where(edges == 255)
    coordinates= list(zip(res[1], res[0]))
    origin = (int(edges.shape[1]/2), edges.shape[0])
    top_point = (edges.shape[1],edges.shape[0])
    for coordinate in coordinates:
        if coordinate[1] < top_point[1] and coordinate[1]>10 and coordinate[0]>10 and coordinate[0]<(edges.shape[1]-10):####
            top_point = coordinate
    cv2.line(edges, origin, top_point, 255, 2)
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    edges_img = weighted_img(edges,img,α=0.8, β=1., γ=0.)
    if abs(top_point[0]-edges.shape[1]/2)<10:
        cv2.putText(edges_img, "Straight", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    elif top_point[0]-edges.shape[1]/2<0:
        cv2.putText(edges_img, "Turn left", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    else:
        cv2.putText(edges_img, "Turn right", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    '''  
    # 找出線段開頭
    interx,intery,left1_x,left1_y,left2_x,left2_y,right1_x,right1_y,right2_x,right2_y=find_point(approx)
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    edges_img = weighted_img(edges,img,α=0.8, β=1., γ=0.)
    point_x,point_y=img.shape[1]/2,img.shape[0]
    turn_dir,angle=Point_T(point_x,point_y,interx,intery)
    left_slope,right_slope=(left2_y-left1_y)/(left2_x-left1_x),(right2_y-right1_y)/(right2_x-right1_x)
    cv2.putText(edges_img, turn_dir, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    cv2.putText(edges_img, str(left_slope), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    cv2.putText(edges_img, str(right_slope), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    cv2.line(edges, (left1_x,left1_y), (left2_x,left2_y), (0,0,255), 10)
    cv2.line(edges, (right1_x,right1_y), (right2_x,right2_y), (0,0,255), 10)
    
    # intersection_method
    '''
    test_images_output,interx,intery=hough_lines(edges,2, np.pi / 180, 100, 25, 25)
    test_images_output = weighted_img(test_images_output, img, α=0.8, β=1., γ=0.)
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    edges_img = weighted_img(edges,test_images_output,α=0.8, β=1., γ=0.)
    point_x,point_y=img.shape[1]/2,img.shape[0]
    turn_dir,angle=Point_T(point_x,point_y,interx,intery)
    cv2.putText(edges_img, turn_dir, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    cv2.putText(edges_img, str(angle), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
    '''
    return edges_img


def main():
    print("start0")
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
    img = cv2.imread(args.img)
    edge_img= pipline(model, img)
    plt.figure(figsize=(10, 5))
    plt.imshow(edge_img)
    plt.savefig("output1.jpg")
    
    # test 
    '''
    cap = cv2.VideoCapture("image/vediotest3.mp4")
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
    '''
    print("OK")

if __name__ == '__main__':
    main()