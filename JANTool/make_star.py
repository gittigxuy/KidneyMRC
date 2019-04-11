import os

import cv2
import numpy as np

from JANTool.img_utils import norm_path, split_path, image_list

# 파일 이름을 받아서 seg가 있는지 판별
# 파일 이름이 세그먼트에 없다면 원본 이미지에 점을 찍고
# seg 그리기
# 그린거
# seg_img 내보냄


def draw_img(src_img, dst_img, x, y):
    h, w = src_img.shape[:2]
    dst_img[y:y+h, x:x+w][src_img == 255] = src_img[src_img == 255]
    return dst_img



def distinguish_bg(image, seg_path, save, save_seg):
    if seg_path is None:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        star = cv2.imread('/home/bong6/Desktop/star.png', cv2.IMREAD_GRAYSCALE)

        h, w = img.shape[:2]
        seg = np.zeros((h, w))


        img = draw_img(star, img, (0 + w - 100)//2, 0)
        seg = draw_img(star, seg,(0 + w - 100)//2, 0)



        # cv2.imshow('fdfa', fake_seg)
        # cv2.imshow("dffd", fake_img)
        # cv2.waitKey(0)
        cv2.imwrite(save, img)
        cv2.imwrite(save_seg, seg)

        print("img", save)
        print("seg", save_seg)
    else:
        img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_path,cv2.IMREAD_GRAYSCALE)
    return img, seg

def main(src_path, dst_path, seg_path, dst_path_seg):

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    if not os.path.exists(dst_path_seg):
        os.makedirs(dst_path_seg)

    mask_dict = {}
    image_dict = {}

    try:
        for (path, dir, files) in os.walk(seg_path):
            for seg_filename in files:
                ext = os.path.splitext(seg_filename)[1]
                if ext == '.jpg' or ext == '.png':
                    mask_path = os.path.join(path, seg_filename)
                    # key: filename, value: image path
                    mask_dict[seg_filename] = mask_path

        for (path, dir, files) in os.walk(src_path):
            for img_filename in files:
                ext = os.path.splitext(img_filename)[1]
                if ext == '.jpg' or ext == '.png':
                    image_path = os.path.join(path, img_filename)
                    # key: filename, value: image path
                    image_dict[img_filename] = image_path

        for file in image_list(src_path):
            _, name, ext = split_path(file)

            save = os.path.join(dst_path, name + ext)
            save_seg = os.path.join(dst_path_seg, name + ext)
            if name+ext not in mask_dict.keys():
                mask_dict[name+ext] = None


            distinguish_bg(image_dict[name+ext], mask_dict[name+ext],save, save_seg)


    except IOError:
        pass  # You can always log it to logger


if __name__ == '__main__':

    src_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404/us/train'
    seg_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404/seg'

    dst_path = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404/new_img'
    dst_path_seg = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/SegMrcnn_kidney_liver_bg_some_400_20190404/new_seg'

    main(src_path, dst_path, seg_path, dst_path_seg)
