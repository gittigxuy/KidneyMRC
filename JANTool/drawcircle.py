# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2

# load the image
image = cv2.imread("/media/bong6/602b5e26-f5c0-421c-b8a5-08c89cd4d4e6/data/yonsei2/dataset/pngs/kidney11/1.2.250.1.204.5.722511.201711140215505718.75.png")
save_path = "/home/bong6/Desktop/sav"
# loop over the alpha transparency values
for alpha in np.arange(0, 1.1, 0.1)[::-1]:
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # draw a red rectangle surrounding Adrian in the image
    # along with the text "PyImageSearch" at the top-left
    # corner
    cv2.rectangle(overlay, (461, 60), (470, 70),
                  (0, 0, 255), -1)
    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    cv2.imshow("Output", output)
    cv2.imwrite(save_path + ".png", output)
    print(save_path)
    cv2.waitKey(0)