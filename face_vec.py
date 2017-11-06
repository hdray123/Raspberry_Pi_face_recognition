# code: utf-8
import facecap
import cv2
from PIL import Image
from aip import AipFace
import time
APP_ID = '*******' 
API_KEY = 'zlfKx7Pra5yuXyHdewy968Yq'
SECRET_KEY = 'puYzTG5D7gN1TfNXzztdmR3H8brNcdqt'

aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)

def face_detect():
    facecap.detect_face()

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def face_to_add(image_peson_path,name,id):
    aipFace.addUser(
        '%s'%id,
        '%s' %name,
        'group1',
        get_file_content(image_peson_path)
    )

def face_to_update(image_person_path,name,id):
    aipFace.updateUser(
        '%s'%id,
        '%s'%name,
        'group1',
        get_file_content(image_person_path)
    )

def face_to_detel(id):
    aipFace.deleteUser('%s'%id)

def get_user_group():
    options = {
        'start': 1,
        'num': 100,
    }
    result=aipFace.getGroupUsers('group1', options)
    return result['result']

def face_ident():
    options = {
        'user_top_num': 1,
        'face_top_num': 1,
    }

    CascadePath = './haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(CascadePath)

    video_capture = cv2.VideoCapture(0)  # Sets video source as default webcam

    while True:
        ret, frame = video_capture.read()  # captures frame by frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(150, 150),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # To draw a rectangle around the detected faces
        for index, (x, y, w, h) in enumerate(faces):
            region = (x, y, x + w, y + h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_image = Image.fromarray(frame)
            cropImg = frame_image.crop(region)
            cropImg.save('%d.jpg' % index)
            result = aipFace.identifyUser(
                'group1',
                get_file_content('%d.jpg'%index),
                options
            )

            print(result)

        cv2.imshow("Video detecting faces", frame)  # Displays resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # To exit q is to be pressed
            break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    face_ident()
