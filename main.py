import os
import sys
import cv2
import glob
import dlib
import imutils
import face_recognition

import numpy as np

from skimage import io 
from pathlib import Path
from imutils import face_utils, resize
from imutils.face_utils import FaceAligner

faces_folder_path = 'faces'
models_folder_oath = 'models'
predictor_path = os.path.join(models_folder_oath, 'shape_predictor_68_face_landmarks.dat')
face_rec_model_path = os.path.join(models_folder_oath, 'dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

def face_reco_with_cam():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    known_face_encodings = []
    known_face_names = []

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")): 
        base = os.path.basename(f)
        vars()[Path(base).stem] = face_recognition.load_image_file(f)
        try:
            known_face_encodings.append(face_recognition.face_encodings(vars()[Path(base).stem])[0])
            known_face_names.append(Path(base).stem)
        except:
            vars()[Path(base).stem] = imutils.resize(vars()[Path(base).stem], width = 1000)
            known_face_encodings.append(face_recognition.face_encodings(vars()[Path(base).stem])[0])
            known_face_names.append(Path(base).stem)
    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        cv2.imshow('Result', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def face_reco(detect_image_path):
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    descriptors = []
    candidate = []

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")): 
        base = os.path.basename(f)
        candidate.append(os.path.splitext(base)[0])
        image = io.imread(f)

        dets = detector(image, 1) 

        for k, d in enumerate(dets):
            shape = sp(image, d)
            face_descriptor = facerec.compute_face_descriptor(image, shape)
            v = np.array(face_descriptor)
            descriptors.append(v)

    detect_image = io.imread(detect_image_path)
    # height, width = detect_image.shape[:2]
    dets = detector(detect_image, 1)
    dist = []

    for k, d in enumerate(dets):
        shape = sp(detect_image, d)
        face_descriptor = facerec.compute_face_descriptor(detect_image, shape)
        d_test = np.array(face_descriptor)
        left = d.left()
        top = d.top()
        right = d.right()
        bottom = d.bottom()
        cv2.rectangle(detect_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(detect_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        for i in descriptors:
            dist_ = np.linalg.norm(i - d_test)
            dist.append(dist_)
    
    
    c_d = dict(zip(candidate, dist))
    cd_sorted = sorted(c_d.items(), key = lambda d:d[1])
    rec_name = cd_sorted[0][0]

    cv2.putText(detect_image, rec_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    img = imutils.resize(detect_image, width = 600)
    img = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_align(image_path):
    predictor = dlib.shape_predictor(predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    image = cv2.imread(image_path)
    # image = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)

if __name__ == '__main__':
    # face_reco_with_cam()
    # face_reco('meow.jpg')
    face_align('faces/arashi87.jpg')