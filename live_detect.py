from model import Model
import helper    
import cv2
from imutils.video import VideoStream
import time

def main(frame, model_mean, config, weights):
    
    '''
    main function for real time age detection
    '''


    model = Model(
        path = '',
        age_model_mean = model_mean,
        age_config = config,
        age_weights = weights,
        face_detector='haarcascade_frontalface_default.xml',
        )
    

    detect_age(model, frame)

def detect_age(model, frame):

    '''
    predicting age of each frame in video stream
    '''

    results = []

    # convert frame to gray
    frame_gray = helper.convert_to_gray(frame)

    # detect faces
    faces = model.detect_face(frame_gray)

    # if no faces, continue to next frame
    if len(faces) == 0: return 


    # get rect
    rects = model.get_rects(faces, frame)
    
    for rect in rects:
        
        # predicting age of face

        age_preds, box = model.detect_age(frame, rect)

        max_pred = age_preds[0].argmax()

        age = model.AGES[max_pred]

        results.append((box, age))

    plot_face(frame, results)

def plot_face(image, results):

    '''
    plot detected face and age
    '''

    msg = 'Face Detected'

    for box, age in results:
        # plot rectangles around face

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 
                        (00, 200, 200), 2)

        # plot age prediction
        cv2.putText(image, f'{msg}:{age}', (box[0],
                                        box[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('image', image)


stream = VideoStream(src=0).start()
time.sleep(1)


while True:
    
    # get current frame of stream 
    frame = stream.read()
    
    # resize frame
    frame = cv2.resize(frame, (400, 400))

    main(
        frame=frame,
        model_mean= (),
        config = "models/gad/age_net.caffemodel",
        weights = "models/gad/age_deploy.prototxt",
        )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    
cv2.destroyAllWindows()
stream.stop()

