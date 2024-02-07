import cv2
import numpy as np
import pytesseract
import imutils

# Thiết lập cấu hình pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Admin\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Tesseract\tesseract.exe'
def detect_license_plate(frame):
    # Chuyển ảnh sang grayscale và làm mờ để loại bỏ nhiễu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện cạnh trong ảnh
    edged = cv2.Canny(blurred, 50, 200)
    
    # Tìm contour trên ảnh cạnh
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Lọc ra contour có diện tích lớn nhất (giả định là biển số xe)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Vẽ contour lớn nhất lên ảnh gốc
    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
    
    # Cắt và lưu lại vùng chứa biển số xe
    x, y, w, h = cv2.boundingRect(largest_contour)
    plate_img = gray[y:y+h, x:x+w]
    
    # Nhận diện văn bản từ ảnh biển số sử dụng Tesseract OCR
    plate_text = pytesseract.image_to_string(plate_img, lang='eng')
    
    # Vẽ văn bản lên khung hình gốc
    cv2.putText(frame, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    # Mở luồng video từ camera
    cap = cv2.VideoCapture(0)
    
    while True:
        # Đọc frame từ luồng video
        ret, frame = cap.read()
        
        if not ret:
            print("Không thể đọc frame từ camera")
            break
        
        # Nhận diện biển số trên frame
        frame_with_plate = detect_license_plate(frame)
        
        # Hiển thị frame chứa biển số
        cv2.imshow('License Plate Detection', frame_with_plate)
        
        # Thoát khỏi vòng lặp nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng luồng video và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()
