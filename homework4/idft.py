from symbol import import_from
from unittest import result
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
def magnitude_phase_split(img):
    # 分离幅度谱与相位谱
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    # 幅度谱
    magnitude_spectrum = np.abs(dft_shift)
    # 相位谱
    phase_spectrum = np.angle(dft_shift)
    return magnitude_spectrum,phase_spectrum

def magnitude_phase_combine(img_m,img_p):
    # 不同图像幅度谱与相位谱结合
    img_mandp = img_m*np.e**(1j*img_p)
    # 图像重构
    img_mandp = np.uint8(np.abs(np.fft.ifft2(img_mandp)))
    img_mandp =img_mandp/np.max(img_mandp)*255
    return img_mandp
 

if __name__ == '__main__':
# 读取图像 主图和纹理图
    img = cv2.imread("Image_processing\homework4\Lena.bmp",0)

# 分离幅度谱与相位谱
    img1_m,img1_p = magnitude_phase_split(img)
    # print(img1_m.shape)
    h1,w1=img1_m.shape[:2]
    h2,w2=img1_p.shape[:2]
    # print(h2,w2)
    zero=np.zeros((h2,w2))
    # print(a)
    A=np.ones((h1,w1))

 
# # 合并不同幅度谱与相位谱
# 1.情况1：
    result1 = magnitude_phase_combine(abs(img1_m),img1_p)
# 2.情况2：
    result2 = magnitude_phase_combine(abs(img1_m),zero) 
# 3.情况3：
    result3 = magnitude_phase_combine(A,img1_p)


# #  获取结果
    cv2.imwrite('Image_processing/homework4/result/original.jpg',img)
    cv2.imwrite('Image_processing/homework4/result/result1.jpg',result1)
    cv2.imwrite('Image_processing/homework4/result/result2.jpg',result2)
    cv2.imwrite('Image_processing/homework4/result/result3.jpg',result3)
