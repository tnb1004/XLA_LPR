import cv2
import imutils
import numpy as np


# Param
max_size = 5000
min_size = 900

# Khởi tạo video stream từ camera hoặc từ file video
# Thay 0 bằng đường dẫn đến file video nếu bạn muốn sử dụng video từ file
video_stream = cv2.VideoCapture(0)

while True:
    # Đọc từng khung hình từ video stream
    ret, frame = video_stream.read()
    
    if not ret:
        print("Không thể đọc video stream hoặc đã đến cuối video.")
        break
    
    # Resize khung hình
    frame = imutils.resize(frame, width=620)
    
    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4 and max_size > cv2.contourArea(c) > min_size:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)

        # Masking the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        # Hiển thị video và kết quả
        cv2.imshow('Video', frame)
        cv2.imshow('License plate', Cropped)

        # Thoát khỏi vòng lặp nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên và đóng cửa sổ
video_stream.release()
cv2.destroyAllWindows()
