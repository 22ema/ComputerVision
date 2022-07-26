import numpy as np
import cv2
import os
import time
import math
from skimage import feature

class RecogImage():

    def __init__(self, image_size=250, crop=25):
        self.prev_hist = None
        self.crop = crop
        self.result = None
        self.transform_img = np.zeros((image_size, image_size))
    
    def np_hist_to_cv(self, np_histogram_output):
        counts = np_histogram_output
        return counts.ravel().astype('float32')


    def recog_human(self, recent_frame):
        self.lbp_run(recent_frame)
        if self.result == None:
            return None, None
        else:
            if self.result <= 0.72:
                return False, self.result
            else:
                return True, self.result

    def get_pixel_else_0(self, l, idx, idy, default=0):
        try:
            return l[idx,idy]
        except IndexError:
            return default

    def LBP(self, img):
        for x in range(0, len(img)):
            for y in range(0, len(img[0])):
                center        = img[x,y]
                top_left      = 1 if center <= self.get_pixel_else_0(img, x-1, y-1) else 0
                top_up        = 2 if center <= self.get_pixel_else_0(img, x, y-1) else 0
                top_right     = 4 if center <= self.get_pixel_else_0(img, x+1, y-1) else 0
                right         = 8 if center <= self.get_pixel_else_0(img, x+1, y ) else 0
                left          = 128 if center <= self.get_pixel_else_0(img, x-1, y ) else 0
                bottom_left   = 64 if center <= self.get_pixel_else_0(img, x-1, y+1) else 0
                bottom_right  = 16 if center <= self.get_pixel_else_0(img, x+1, y+1) else 0
                bottom_down   = 32 if center <= self.get_pixel_else_0(img, x,   y+1 ) else 0
                values = [top_left, top_up, top_right, right, bottom_right, bottom_down, bottom_left, left]

                res = sum(values)
                self.transform_img.itemset((x,y), res)
        return self.transform_img

    def lbp_run(self, recent_frame):
        gray = cv2.cvtColor(recent_frame, cv2.COLOR_BGR2GRAY)
        resize_frame = cv2.resize(gray, (250, 250))
        hist=list()
        numPoints = 8
        radius = 1
        lbp = feature.local_binary_pattern(resize_frame, numPoints, radius, method="default")
        ## you use your alogrithm about LBP.
        # lbp = self.LBP(resize_frame[y-self.crop:y,x-self.crop:x])
        for x in range(self.crop, 251, self.crop):
            for y in range(self.crop, 251, self.crop):
                for i in np.histogram(lbp[x-25:x, y-25:y].flatten(),10,[0,255])[0]:
                    hist.append(i/625)
        if self.prev_hist == None:
            self.result = None
        else:
            self.result = cv2.compareHist(self.np_hist_to_cv(np.asarray(self.prev_hist)), self.np_hist_to_cv(np.asarray(hist)), cv2.HISTCMP_CORREL)
        self.prev_hist = hist

# exmaple code
if __name__ == "__main__":
    path = "../../../dataset/test/sinhan_atm_test/head_only_data"
    # path = "../../../dataset/test/sinhan_atm_test/1"
    dir_list= sorted(os.listdir(path), key=lambda x : int(x.split('.')[0]))
    prev_hist = list()
    recog_human = RecogImage()
    for file_index in range(0, len(dir_list)):
        recent_frame_path = os.path.join(path, dir_list[file_index])
        # 이미지
        recent_frame = cv2.imread(recent_frame_path)
        recog_change, result_hist = recog_human.recog_human(recent_frame)
        if result_hist == None:
            print("Not exist prev frame")
        else:
            if recog_change == False:
                print(dir_list[file_index], "person_change", result_hist)
            else:
                print(dir_list[file_index], "person_not_change", result_hist)


