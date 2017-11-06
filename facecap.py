import cv2
import sys
from PIL import Image
def detect_face():
    CascadePath ='./haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(CascadePath)

    video_capture = cv2.VideoCapture(0)  #Sets video source as default webcam

    while True:
        ret, frame = video_capture.read() #captures frame by frame

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
         )

    #To draw a rectangle around the detected faces
        for index,(x, y, w, h)   in  enumerate(faces):

            region = (x, y, x + w, y + h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame_image=Image.fromarray(frame)
            cropImg = frame_image.crop(region)
            cropImg.save('%d.jpg' %index)

        cv2.imshow("Video detecting faces" ,frame)  #Displays resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):   #To exit q is to be pressed
            break

    video_capture.release()
    cv2.destroyAllWindows()
