from PIL import Image, ImageFilter, ImageOps
import torch
import cv2 as cv
import numpy as np
import json
import os.path as osp
import os
import sys, time
from collections import namedtuple

gts_gray_path = './imgpre'  # 灰度图的路径（原图）
gts_color_path = './imgprecolor'  # 转换之后的彩色图路径
Cls = namedtuple('cls', ['name', 'id', 'color'])
Clss = [
    Cls('road', 0, (128, 64, 128)),  # 这些标签颜色来自于github上面的cityscapesscriptshelperslabel.py
    Cls('sidewalk', 1, (244, 35, 232)),
    Cls('building', 2, (70, 70, 70)),
    Cls('wall', 3, (102, 102, 156)),
    Cls('fence', 4, (190, 153, 153)),
    Cls('pole', 5, (153, 153, 153)),
    Cls('traffic light', 6, (250, 170, 30)),
    Cls('traffic sign', 7, (220, 220, 0)),
    Cls('vegetation', 8, (107, 142, 35)),
    Cls('terrain', 9, (152, 251, 152)),
    Cls('sky', 10, (70, 130, 180)),
    Cls('person', 11, (220, 20, 60)),
    Cls('rider', 12, (255, 0, 0)),
    Cls('car', 13, (0, 0, 142)),
    Cls('truck', 14, (0, 0, 70)),
    Cls('bus', 15, (0, 60, 100)),
    Cls('train', 16, (0, 80, 100)),
    Cls('motorcycle', 17, (0, 0, 230)),
    Cls('bicycle', 18, (119, 11, 32))
]

def gray_color(color_dict, gray_path=gts_gray_path, color_path=gts_color_path):
    # #
    #     '''
    #     swift gray image to color, by color mapping relationship
    #     param color_dictcolor mapping relationship, dict format
    #     param gray_pathgray imgs path
    #     param color_pathcolor imgs path
    #     return
    #     '''
    pass
    t1 = time.time()
    gt_list = os.listdir(gray_path)
    for index, gt_name in enumerate(gt_list):
        gt_gray_path = os.path.join(gray_path, gt_name)
        gt_color_path = os.path.join(color_path, gt_name)
        gt_gray = cv.imread(gt_gray_path, cv.IMREAD_GRAYSCALE)
        assert len(gt_gray.shape) == 2  # make sure gt_gray is 1band
        gt_color = matrix_mapping(color_dict, gt_gray)
        # endregion

        gt_color = cv.cvtColor(gt_color, cv.COLOR_RGB2BGR)
        cv.imwrite(gt_color_path, gt_color, )
        process_show(index + 1, len(gt_list))
    print(time.time() - t1)


def color_gray(color_dict, color_path=gts_color_path, gray_path=gts_gray_path):
        # '''
        # swift color image to gray, by color mapping relationship
        # param color_dictcolor mapping relationship, dict format
        # param gray_pathgray imgs path
        # param color_pathcolor imgs path
        # return
        # '''
        gray_dict = {}
        for k, v in color_dict.items():
            gray_dict[v] = k
        t1 = time.time()
        gt_list = os.listdir(color_path)
        for index, gt_name in enumerate(gt_list):
            gt_gray_path = os.path.join(gray_path, gt_name)
            gt_color_path = os.path.join(color_path, gt_name)
            color_array = cv.imread(gt_color_path, cv.IMREAD_COLOR)
            assert len(color_array.shape) == 3

            gt_gray = np.zeros((color_array.shape[0], color_array.shape[1]), np.uint8)
            b, g, r = cv.split(color_array)
            color_array = np.array([r, g, b])
            for cls_color, cls_index in gray_dict.items():
                cls_pos = arrays_jd(color_array, cls_color)
                gt_gray[cls_pos] = cls_index

            cv.imwrite(gt_gray_path, gt_gray)
            process_show(index + 1, len(gt_list))

        print(time.time() - t1)


def arrays_jd(arrays, cond_nums):
    r = arrays[0] == cond_nums[0]
    g = arrays[1] == cond_nums[1]
    b = arrays[2] == cond_nums[2]
    return r & g & b


def matrix_mapping(color_dict, gt):
    colorize = np.zeros([len(color_dict), 3], 'uint8')
    for cls, color in color_dict.items():
        colorize[cls,] = list(color)
    ims = colorize[gt,]
    ims = ims.reshape([gt.shape[0], gt.shape[1], 3])
    return ims


def nt_dic(nt=Clss):
    # '''
    # swift nametuple to color dict
    # param nt nametuple
    # return
    # '''
    pass
    color_dict = {}
    for cls in nt:
        color_dict[cls.id] = cls.color
    return color_dict


def process_show(num, nums, pre_fix='', suf_fix=''):
    '''
    auxiliary function, print work progress
    param num
    param nums
    param pre_fix
    param suf_fix
    return
    '''
    rate = num
    nums
    ratenum = round(rate, 3)*100
    bar = 'r%s %g%g [%s%s]%.1f%% %s' %\
        (pre_fix, num, nums, '#'*(int(ratenum) // 5), '_'*(20 - (int(ratenum)//5)), ratenum, suf_fix)
    sys.stdout.write(bar)
    sys.stdout.flush()




class ImageProc():
    def __init__(self, *args, **kwargs):
        print('1')

    # def drawBoudary(blur=26,alpha=1.0):
    #     img = Image.open('/home/lab/xyy/STDC-Seg/nets/1.png')
    #     out = img.resize((1024, 512),
    #                      Image.ANTIALIAS)  # Image.NEAREST低质量；Image.BILINEAR双线性；Image.BICUBIC 三次样条插值；Image.ANTIALIAS高质量
    #     img1 = img.convert('L') #图片转换成灰色
    #     img2 = img1.copy() #复制图片
    #     img2 = ImageOps.invert(img2) #实现二值图像的黑白翻转
    #    for i in range(26):# blur模糊度
    #     img2 = img2.filter(26) #ImageFilter.BLUR为模糊滤波，处理之后的图像会整体变得模糊。
    #                                                #ImageFilter.CONTOUR为轮廓滤波，将图像中的轮廓信息全部提取出来。
    #     width, height = img1.size
    #     for x in range(width):
    #         for y in range( height):
    #             a = img1.getpixel((x, y))
    #             b = img2.getpixel((x, y))
    #             #img1.putpixel((x,y),dodge())
    #             img1.putpixel((x, y),min(int(a*255/(256-b*1)),255))#在指定位置上放一像素
    #
    #     img1.show()
    #     img1.save('11.jpg')

    def drawBoudary2(self,a):
        img_name = '/home/lab/xyy/STDC-Seg/nets/1.png'
        img = cv.imread(img_name)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
        h = img.shape[0]
        w = img.shape[1]
        for i in range(h):
            for j in range(w):
                if gray_img[i][j] > 127:
                    bin_img[i][j] = 255
                else:
                    bin_img[i][j] = 0
        cv.imwrite('/home/lab/xyy/STDC-Seg/nets/2.png',bin_img)
        #cv.imshow("主窗口",bin_img)
        imgs = Image.open('/home/lab/xyy/STDC-Seg/nets/2.png')
        imgs.show()
        return bin_img
    def drawBoudary3(self,a):
        #folder_label = '/content/data/label/'
        #folder_img = '/content/data/img/'
        #folder_path = '/content/data/'
        #file_list = os.listdir(folder_path)

        #with open('../cityscapes_info.json', 'r') as fr:
        #    labels_info = json.load(fr)
        #self.lb_map = {el['id']: el['trainId'] for el in labels_info}  # color,:el['color']

        #label_dict = {'flat': 0, 'human': 1, 'vehicle': 2, 'construction': 3, 'object': 4, 'nature': 5, 'sky': 6}
        #color = {'flat': (255, 0, 0), 'human': (255, 128, 0), 'vehicle': (255, 255, 0), 'construction': (0, 255, 0),
        #         'object': (0, 255, 255), 'nature': (0, 0, 255), 'sky': (128, 0, 255)}
        #if len(os.listdir('/content/data/img')) <= 20:
        #    for i in tqdm(file_list):
        #        num = 0
        #        name, type_ = os.path.splitext(i)
        #        if type_ == '.json':
        #            file_path = os.path.join(folder_path, i)
        file = open('/home/lab/xyy/STDC-Seg/nets/test.json').read()
        file = json.loads(file)
        imgHeight = file['imgHeight']  # 图像高
        imgWidth = file['imgWidth']  # 图像宽
        # img = np.zeros((imgHeight, imgWidth, 3), np.uint8)  # 二十四位彩图
        img = np.zeros((imgHeight, imgWidth), np.uint8)  # 八位图
        img.fill(0)
        # print(img.shape)
        objects = file['objects']
        for j in objects:
            label = j['label']
            contours = np.array(j['polygon'])
            cv.drawContours(img, [contours], -1, 255, 2)# 255, -1
            #save_path = os.path.join(folder_label, str(name) + '.png')
        cv.imwrite('/home/lab/xyy/STDC-Seg/nets/test.png', img)
        imgs = Image.open('/home/lab/xyy/STDC-Seg/nets/test.png')
        imgs.show()
        return img

if __name__ == "__main__":
    #---------------------------------------
    #draw  Boudary form .json
    #myImagePrco = ImageProc()
    #a = 1
    #myImagePrco.drawBoudary3(a)
    # ---------------------------------------

    # ---gray preimg to color img------------------------------------
   
    color_dict = nt_dic()
    gray_color(color_dict)
    # color_gray(color_dict)





