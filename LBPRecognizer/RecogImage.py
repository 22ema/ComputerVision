import numpy as np
import cv2
import os
import time
import math
class RecogImage():

    def __init__(self, crop=25):
        self.prev_hist = None
        self.result_hist = None
        self.crop = crop
        self.transform_img = np.zeros((crop, crop))


    def recog_human(self, recent_frame):
        self.lbp_run(recent_frame)
        if self.result_hist == None:
            return None, None
        else:
            result = sum(self.result_hist)
            if round(result) >= 10:
                return False, result
            else:
                return True, result

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
    # def Adaptive_LBP(self, img):
    #     for x in range(0, len(img)):
    #         for y in range(0, len(img[0])):
    #             center        = img[x,y]
    #             top_left      = self.get_pixel_else_0(img, x-1, y-1) - int(center)
    #             top_up        = self.get_pixel_else_0(img, x, y-1) - int(center)
    #             top_right     = self.get_pixel_else_0(img, x+1, y-1) - int(center)
    #             right         = self.get_pixel_else_0(img, x+1, y ) - int(center)
    #             left          = self.get_pixel_else_0(img, x-1, y ) - int(center)
    #             bottom_left   = self.get_pixel_else_0(img, x-1, y+1) - int(center)
    #             bottom_right  = self.get_pixel_else_0(img, x+1, y+1) - int(center)
    #             bottom_down   = self.get_pixel_else_0(img, x,   y+1 ) - int(center)
    #             w = ((top_left + top_up + top_right + right + left + bottom_left + bottom_right + bottom_down) / 9) + math.sqrt(max(map(max, img))-min(map(min, img)))
    #             values = self.thresholded(w, [top_left, top_up, top_right, right, bottom_right,
    #                                         bottom_down, bottom_left, left])

    #             weights = [1, 2, 4, 8, 16, 32, 64, 128]
    #             res = 0
    #             for a in range(0, len(values)):
    #                 res += weights[a] * values[a]

    #             self.transform_img.itemset((x,y), res)

    def lbp_run(self, recent_frame):
        gray = cv2.cvtColor(recent_frame, cv2.COLOR_BGR2GRAY)
        resize_frame = cv2.resize(gray, (250, 250))
        hist=list()
        for x in range(25, 251, 25):
            for y in range(25, 251, 25):
                self.LBP(resize_frame[y-25:y,x-25:x])
                for i in np.histogram(self.transform_img.flatten(),10,[0,255])[0]:
                    hist.append(i/625)
        if self.prev_hist == None:
            self.result_hist = None
        else:
            # self.result_hist = [((self.prev_hist[i]-hist[i])**2)/(self.prev_hist[i]+hist[i]) for i in range(0, len(hist))]
            self.result_hist = [((self.prev_hist[i]-hist[i])**2) for i in range(0, len(hist))]
        self.prev_hist = hist

# exmaple code
if __name__ == "__main__":
    path = "../../../dataset/test/sinhan_atm_test/head_only_data"
    dir_list= sorted(os.listdir(path), key=lambda x : int(x.split('.')[0]))
    prev_hist = list()
    recog_human = RecogImage()
    for file_index in range(0, len(dir_list)):
        recent_frame_path = os.path.join(path, dir_list[file_index])
        # 이미지
        recent_frame = cv2.imread(recent_frame_path)
        recog_change, result_hist = recog_human.recog_human(recent_frame)
        break
        if result_hist == None:
            print("Not exist prev frame")
        else:
            if recog_change == False:
                print(dir_list[file_index], "person_change", result_hist)
            # else:
            #     print(dir_list[file_index], "person_not_change", result_hist)


