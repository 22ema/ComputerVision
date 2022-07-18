import cv2
import numpy as np

def skindetector(img):
    '''
    parameter :
    - img : image frame
    return :
    - YCrCn_result : image frame
    '''
    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    return YCrCb_result

def luminence_Enhansment(src):
    '''
    parameter :
    - src : image frame
    return :
    - new_hsv: image frame
    '''
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    V_N = V/255
    img_size = len(V[0]) * len(V)
    hist, bins = np.histogram(V.flatten(), 255,[0,255])
    cdf = hist.cumsum()
    if cdf[50] > img_size * 0.1:
        z = 0
    elif img_size * 0.1 > cdf[150]:
        z = 1
    elif cdf[50] <= img_size * 0.1 <= cdf[150]:
        z = ((img_size*0.1)-50)/100
    for i in range(0, len(H)):
        for j in range(0, len(H[0])):
            V_I = V_N[i, j]
            V_N[i, j] = ((V_I**(0.75*z+0.25))+(0.4*(1-z)*(1-V_I))+V_I**(2-z))*0.5
            V_N[i, j] = int(V_N[i, j]*255)
    V_N = np.array(V_N, dtype=S.dtype)
    new_hsv = cv2.merge([H, S, V_N])
    new_hsv = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
    return new_hsv