import cv2
from pytesseract import pytesseract
from PIL import Image


cam = cv2.VideoCapture(0)
while True:
    #doc tung frame
    ret, frame = cam.read()
    #chuyen sang anh xam
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #nhung file nhan dien bien so xe
    license_plate_detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

    #khai bao vung bien so xe khi duoc file nhan dang phat hien
    detections  =  license_plate_detector.detectMultiScale(framegray, scaleFactor=1.05, minNeighbors=3)

    #tao 4 toa do cho vung bien so xe duoc phat hien
    for(x, y, w, h) in detections:
        #ve hinh chu nhat quanh vung bien so xe
        cv2.rectangle(framegray,  (x,y), (x+w, y+h), (0, 255, 255), 2) 
        #inn chu canh bien so xe
        cv2.putText(framegray, "Bien so xe", (x-20, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255,255),2)
        
        #tao 1 khung chi chua hinh anh bien so nhan dien duoc
        license_plate = framegray[y:y + h, x:x + w]
        #chuyen sang anh xam
        gray = cv2.cvtColor(framegray, cv2.COLOR_BGR2GRAY)
        #in khung bien so xe nhan duoc
        cv2.imshow("Bien so xe", gray)
        key = cv2.waitKey(1)
        if key==ord('s'):
            #luu anh vao thu muc images
            cv2.imwrite('images/licenseplate.jpg',gray)
    if cv2.waitKey(1)==ord('q'):
        break
    
    #show khung Cam(hinh anh khi mo camera cua may tinh)
    cv2.imshow("Cam", framegray)

cam.release()
cv2.destroyAllWindows()



