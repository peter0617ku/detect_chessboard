import cv2
import matplotlib.pyplot as plt
import numpy as np

def reshape_list(lst, rows, cols):
    if rows * cols != len(lst):
        raise ValueError("無法將列表轉換為指定的行和列數。")
    return [lst[i:i + cols] for i in range(0, len(lst), cols)]

def find_position(image_path):
    # 讀取圖像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 定義紅色範圍 (這裡的數值需要根據你的圖片調整)
    lower_red = np.array([200, 0, 0])
    upper_red = np.array([255, 50, 50])

    # 創建遮罩
    mask = cv2.inRange(img, lower_red, upper_red)

    # 查找輪廓
    _, binary_image = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chess_position = []
    
    # 計算每個輪廓的質心
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            chess_position.append([cx, cy])

    # 先按照 x 的值進行排序
    chess_position.sort(key=lambda pair: pair[0])

    # 再按照 y 的值進行排序
    chess_position.sort(key=lambda pair: pair[1])

    # 顯示找到的遮罩
    plt.imshow(mask)
    plt.show()

    return chess_position

def read_chess(chess_img, chess_position):
    img = cv2.imread(chess_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    chess = []
    for pos in chess_position:
        x, y = pos
        r, g, b = img[y, x]  # 注意，這裡的索引是 BGR，而不是 RGB
        # print(f"位置 ({x}, {y}) 的 RGB 值：R={r}, G={g}, B={b}")
        if r > b and r > 150:
            # print("red")
            chess.append("r")
        elif b > r and b > 150:
            # print("blue")
            chess.append("b")
        else:
            # print("??")
            chess.append("?")
        
        
    new_list = reshape_list(chess, 6, 7)    

    return new_list

def show_position(image_path, chess_position):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for position in chess_position:
        x, y = position
        img = cv2.circle(img, (x, y), radius=20, color=(0, 0, 255), thickness=-1)

    plt.imshow(img)
    plt.show()

def main():
    image_path = "calibrate_image/calibrate.jpg"
    chess_position = find_position(image_path)

    print(chess_position)

    show_position(image_path, chess_position)

    chess_img = "test_image/test_01.jpg"
    chessboard = read_chess(chess_img, chess_position)

    for single_list in chessboard:
        print(single_list)

if __name__ == "__main__":
    main()
