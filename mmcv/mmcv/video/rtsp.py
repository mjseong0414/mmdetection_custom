import cv2

url = "rtsp://192.168.0.23:8554/live1"

cap = cv2.VideoCapture(url)

while True:
    _, frame = cap.read()
    cv2.imshow("RTSP", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllwindows()