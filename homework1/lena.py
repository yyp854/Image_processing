# 1.生成Lena图片不同空间分辨率
# 用降采样来模拟降低空间分辨率的效果
import cv2
import numpy as np
import matplotlib.pyplot as plt

def down_sample(image):
    height, width = image.shape[:2]
    dst = np.zeros([height//2, width//2])
    dst = image[::2, ::2]
    return dst

img = cv2.imread('./homework1/Lena.jpg', 1)
# cv2.imshow("img",img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_2 = down_sample(img)
img_3 = down_sample(img_2)
img_4 = down_sample(img_3)

plt.figure(figsize=(15, 18))
plt.subplot(221), plt.imshow(img, 'gray'), plt.xticks([]), plt.yticks([])
cv2.imwrite("./homework1/result/img.jpg",img)
plt.subplot(222), plt.imshow(img_2, 'gray'), plt.xticks([]), plt.yticks([])
cv2.imwrite("./homework1/result/img_2.jpg",img_2)
plt.subplot(223), plt.imshow(img_3, 'gray'), plt.xticks([]), plt.yticks([])
cv2.imwrite("./homework1/result/img_3.jpg",img_3)
plt.subplot(224), plt.imshow(img_4, 'gray'), plt.xticks([]), plt.yticks([])
cv2.imwrite("./homework1/result/img_4.jpg",img_4)
# plt.tight_layout()
# plt.show()

# 2.不同灰度分辨率
# 改变灰度值以实现灰度级的不同

# fig = plt.figure(figsize=(13, 26))
# for i in range(8):
#     ax = fig.add_subplot(4, 2, i+1)
#     if i < 7:
#         dst = np.uint(img * (2**(8 - i) - 1))
#     else:
#         dst = np.uint(img * (2))
#     ax.imshow(dst, 'gray')
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.tight_layout()
# plt.show()
