import cv2

cam = cv2.VideoCapture(0)

while True:
    # doc tung frame
    ret, frame = cam.read()
    
    # sang muc xam
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    license_plate_detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

    #khai bao vung bien so xe khi duoc file nhan dang phat hien
    detections = license_plate_detector.detectMultiScale(framegray, scaleFactor=1.05, minNeighbors=3)

    #tao 4 toa do cho vung bien so xe duoc phat hien
    for (x, y, w, h) in detections:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)       
        
        cv2.putText(frame, "License Plate", (x-20, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)
        
         #tao 1 khung chi chua hinh anh bien so nhan dien duoc
        license_plate = framegray[y:y+h, x:x+w]
              
        cv2.imshow("License Plate", license_plate)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('License_plate/numberplate.jpg', license_plate)

    cv2.imshow("Cam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
