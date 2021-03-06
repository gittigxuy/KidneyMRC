import os

import cv2
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from demo.predictor_kidney import KidneyDemo
from maskrcnn_benchmark.config import cfg


def load(path):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(path).convert("RGB")

    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.waitforbuttonpress()


def norm_path(path, makedirs=False):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if makedirs and not os.path.exists(path):
        os.makedirs(path)

    return path


def fileName(filePath):
    dir, fileNameExt = os.path.split(filePath)

    return fileNameExt


def image_list(path, exts=['.png', '.jpg'], recursive=True, followlinks=True):
    path = norm_path(path)

    l = list()
    if recursive:
        for (root, dirs, files) in os.walk(path, followlinks=followlinks):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext.lower() not in exts:
                    continue

                l.append(os.path.join(root, file))
    else:
        for fileDir in os.listdir(path):
            if os.path.isfile(os.path.join(path, fileDir)):
                file = fileDir
            else:
                continue

            name, ext = os.path.splitext(file)
            if ext.lower() not in exts:
                continue

            l.append(os.path.join(path, file))
    return l


def resize_aspect_ratio(img, sizeWH):
    resize_info = {'original_size': [None, None],
                   'pad_top_bottom_left_right': [None, None, None, None],
                   'windowXYWH': [None, None, None, None]}

    dw, dh = sizeWH
    h, w = img.shape[:2]
    resize_info['original_size'] = [w, h]
    ratio = min([sizeWH[1] / h, sizeWH[0] / w])

    nw, nh = (int(w * ratio), int(h * ratio))
    img = cv2.resize(img, (nw, nh))

    padL, padT = (dw - nw) // 2, (dh - nh) // 2
    padR, padB = dw - nw - padL, dh - nh - padT
    resize_info['pad_top_bottom_left_right'] = [padT, padB, padL, padR]
    resize_info['windowXYWH'] = [padL, padT, nw, nh]

    img = cv2.copyMakeBorder(img, padT, padB, padL, padR, borderType=cv2.BORDER_CONSTANT, value=0)

    return img, resize_info


def resize_restore(img, resize_info):
    xw, yw, ww, hw = resize_info['windowXYWH']
    img = img[yw:yw + hw, xw:xw + ww]

    w, h = resize_info['original_size']
    img = cv2.resize(img, (w, h))

    return img

def evaluate(correct_label,label):
    average = 0
    correct = 0
    wrong = 0

    if correct_label==label:
        correct += 1
    else:
        wrong += 1
    sum = correct+wrong
    average = (correct/sum)
    print('sum',sum)
    print('average', average)
    return average


class Acc_meter:
    def __init__(self):
        self.eval_list = []

    def update(self, correct_label, evel_label):
        self.eval_list.append([correct_label, evel_label])

    def get_acc(self):
        correct = 0
        wrong = 0

        for correct_label, evel_label in self.eval_list:
            if correct_label == evel_label:
                correct += 1
            else:
                wrong += 1

        return correct / (correct + wrong)

    def get_class_acc(self):

        correct_wrong_dict = dict()
        for correct_label, evel_label in self.eval_list:
            if correct_label not in correct_wrong_dict:
                correct_wrong_dict[correct_label] = [0, 0]

            if correct_label == evel_label:
                correct_wrong_dict[correct_label][0] += 1
            else:
                correct_wrong_dict[correct_label][1] += 1

        label_acc = []

        labels = list(correct_wrong_dict.keys())
        labels.sort()

        for label in labels:
            correct, wrong = correct_wrong_dict[label]
            acc = correct / (correct + wrong)
            label_acc.append([label, acc])

        return label_acc


def main(dir_path=None, config_file=None, model_file=None, save_dir=None):
    dir_path = norm_path(dir_path) if dir_path else None
    config_file = norm_path(config_file) if config_file else None
    model_file = norm_path(model_file) if model_file else None
    save_dir = norm_path(save_dir, makedirs=True) if save_dir else None
    save_crop_dir = norm_path(os.path.join(save_dir, 'crop'), makedirs=True) if save_dir else None
    save_mask_dir = norm_path(os.path.join(save_dir, 'mask'), makedirs=True) if save_dir else None

    print('paths', save_dir, save_crop_dir, save_mask_dir)

    # this makes our figures bigger
    pylab.rcParams['figure.figsize'] = 20, 12
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    cfg.merge_from_list(["MODEL.WEIGHT", model_file])

    kidney_demo = KidneyDemo(
        cfg,
        min_image_size=512,
        confidence_threshold=0.3,
    )

    class_names = ['', 'AKI', 'CKD', 'normal']

    acc_meter = Acc_meter()
    for filePath in image_list(dir_path):

        print('file_path', filePath)
        img = cv2.imread(filePath, cv2.IMREAD_COLOR)

        # resize image for input size of model
        img, resize_info = resize_aspect_ratio(img, (512, 512))

        #find label high score
        result, crops, masks, labels = kidney_demo.detection(img)
        top_label = labels[0]

        #find diagnosis in directory, it will be used as an correct answer
        path = os.path.split(filePath)[0]
        diagnosis, accno = path.split('/')[-2:]
        #convert diagnosis to int(because labels is integer)

        diag_class_no = class_names.index(diagnosis)

        # print(diagnosis)
        # print(top_label)

        # evaluate(diagnosis,top_label)
        acc_meter.update(diag_class_no, top_label)

        # restore size of image to original size
        result = resize_restore(result, resize_info)

        # save result image
        if save_dir:
            save_file = os.path.join(save_dir, fileName(filePath))
            cv2.imwrite(save_file, result)
        else:
            imshow(result)

        # if found object, make corp and mask image
        if len(labels) > 0:
            for crop, mask, label in zip(crops, masks, labels):

                if save_crop_dir:
                    save_file = os.path.join(save_crop_dir, fileName(filePath))
                    crop = resize_restore(crop, resize_info)
                    cv2.imwrite(save_file, crop)

                if save_mask_dir:
                    save_file = os.path.join(save_mask_dir, fileName(filePath))
                    mask = resize_restore(mask, resize_info)
                    cv2.imwrite(save_file, mask)


    print('acc:{:3.2f}'.format(acc_meter.get_acc()))
    for class_num, class_acc in acc_meter.get_class_acc():
        print('class {} acc {:3.2f}'.format(class_num, class_acc * 100))

if __name__ == '__main__':

    # config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x_kidney.yaml"
    # dir_path = "/home/bong07/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/val"
    # model_file = "/home/bong07/lib/robin_mrcnn/checkpoint/20190121-164312/model_0165000.pth"
    # save_dir = '../result/20190118-175358_model_0165000'

    # config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x_kidney_using_pretrained_model.yaml"
    # dir_path = "/home/bong07/data/yonsei2/dataset/US_isangmi_400+100+1200_withExcluded/val"
    # model_file = "/home/bong07/lib/robin_mrcnn/checkpoint/kidney_using_pretrained_model/model_0180000.pth"
    # save_dir = '../result/kidney_using_pretrained_model_model_0180000'

    # all save path
    config_file = "../configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x_kidney_using_pretrained_model.yaml"
    dir_path = "/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/US_kidney_original_one/val"
    model_file = "/home/bong6/lib/KidneyProject/checkpoint/kidney_using_pretrained_model/model_0010500.pth"
    save_dir = '/home/bong6/kidney_result'



    main(dir_path=dir_path, config_file=config_file, model_file=model_file, save_dir=save_dir)
