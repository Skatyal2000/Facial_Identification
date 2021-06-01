import cv2

face_c = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_c = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_c = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(grey,frame):
    face = face_c.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_c.detectMultiScale(roi_grey,1.1,7)
        smile = smile_c.detectMultiScale(roi_grey,1.6,20)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        for (ex,ey,ew,eh) in smile:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    return frame

vc = cv2.VideoCapture(0)
while True:
    _,frame = vc.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(grey,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()