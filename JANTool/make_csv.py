import os
import numpy as np
import cv2
import csv
#change csv as kidney
#read bbox image and crop image data
image_data_dir = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegMrcn_04_19/us'
mask_data_dir = '/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/SegMrcn_04_19/seg'
result_dir = '/home/bong6/data/mrcnn_cer/liver_kideny.csv'


#make folder
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

padding_size = 0

def find_bounding_square(image, padding=0):
    image = image.astype(np.uint8)

    # get shape
    height, width = image.shape
    x1 = width
    y1 = height
    x2 = 0.0
    y2 = 0.0

    _, contours, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if x < x1: x1 = x
        if y < y1: y1 = y
        if (x + w) > x2: x2 = x + w
        if (y + h) > y2: y2 = y + h

    w, h = x2 - x1, y2 - y1

    x, y, w_padded, h_padded = (x1 - padding, y1 - padding, w + padding * 2, h + padding * 2)

    # correct figures (Problems arising from padding, for example -1)
    x = x if x > 0 else 0
    y = y if y > 0 else 0

    return x, y, w_padded, h_padded


#read imagefiles

image_dict = {}
mask_dict = {}

#read image data dir
for (path, dir, files) in os.walk(image_data_dir):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        if ext == '.png' or ext == '.jpg':
            image_path = os.path.join(path, filename)
            # key: filename, value: image path
            image_dict[filename] = image_path



# read mask data dir

for (path, dir, files) in os.walk(mask_data_dir):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        if ext == '.jpg' or ext == '.png':
            mask_path = os.path.join(path, filename)
            # key: filename, value: image path
            mask_dict[filename] = mask_path




#open csv file
with open('output_resize_2type.csv', 'w', encoding='utf-8', newline='') as f:
    fildnames = ['filename', 'x', 'y', 'w', 'h', 'Type']
    wr = csv.writer(f)
    wr.writerow(fildnames)
    for id, image_path in image_dict.items():
        ext = os.path.splitext(id)[1]
        if ext == '.jpg' or ext == '.png':


            if id not in mask_dict:
                continue

            #이미지에서 좌표를 뽑아낸다
            input_image = cv2.imread(image_path)
            input_mask_image = cv2.imread(os.path.join(mask_dict[id]), cv2.IMREAD_GRAYSCALE)
            # input_mask_image = cv2.cvtColor(input_mask_image, cv2.COLOR_BGR2GRAY)

            x, y, w, h = find_bounding_square(input_mask_image, padding=padding_size)

            # for debug
            result_image = input_image[int(y):int(y + h), int(x):int(x + w)]

            kidneylist=['kidney', 'liver']

            # 타입별로 나누도록
            if kidneylist[0] in mask_dict:
                type = kidneylist[0]
            else:
                type = kidneylist[1]

            id = id.replace(ext, '')

            wr.writerow([id, x, y, w, h, type])
            # print([id, x, y, w, h, type])