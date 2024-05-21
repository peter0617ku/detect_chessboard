import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("20240521_215954.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 定義紅色範圍 (這裡的數值需要根據你的圖片調整)
lower_red = np.array([200, 0, 0])
upper_red = np.array([255, 50, 50])

# 創建遮罩
mask = cv2.inRange(img, lower_red, upper_red)

_, binary_image = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# 查找輪廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
position = []
# 計算每個輪廓的質心
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"Contour center: ({cx}, {cy})")
        position.append([cx,cy])

# 先按照 x 的值進行排序
position.sort(key=lambda pair: pair[0])

# 再按照 y 的值進行排序
position.sort(key=lambda pair: pair[1])
print(position)
plt.imshow(mask)
plt.show()
