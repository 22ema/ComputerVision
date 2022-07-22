
from . import SkinDetector

def counting_black(img):
    '''
    counting 0 value pixel for counting black pixels
    '''
    count = 0
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if img[i, j] == 0:
                count += 1
    return count


def call_decision(img):
    '''
    parameters :
    img = image frame

    return :
    left or right = black pixels / all pixels of crop_image
    'right_hand' or 'left_hand' (str)
    '''
    height, width = img.shape[0], img.shape[1]
    Enhan_img = SkinDetector.luminence_Enhansment(img)
    skin_img = SkinDetector.skindetector(Enhan_img)
    l_img = skin_img[height-(height//2): height, 0:width-(width//2)]
    r_img = skin_img[height-(height//2): height, width-(width//2):width]
    l_pixels = counting_black(l_img)
    r_pixels = counting_black(r_img)
    left = l_pixels/(l_img.shape[0]*l_img.shape[1])
    right = r_pixels/(r_img.shape[0]*r_img.shape[1])
    if l_pixels > r_pixels:
        return left, 'right_hand'
    elif l_pixels <= r_pixels:
        return right, 'left_hand'

## example
# if __name__ == "__main__":
#     path = "../../dataset/test/sinhan_atm_test/1/9400.jpg"
#     img = cv2.imread(path)
#     ratio, result = call_decision(img)
#     print(result)