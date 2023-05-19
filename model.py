import dlib
import cv2
from matplotlib import pyplot as plt
import time
import pandas as pd

import helper

OUTPUT_FLAG = True

class Model():
    
    def __init__(
        self, 
        path,
        age_model_mean, 
        age_weights, 
        age_config,
        face_detector = 'models/mmod_human_face_detector.dat',
    ):
        
        '''
        initialize detection models and parameters
        '''
        
        # age labels
        self.AGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

        # image dataset base path
        self.base_path = path

        # face detector parameters
        self.face_detector = face_detector

        # age detector parameters
        self.age_model_mean = age_model_mean
        self.age_weights = age_weights
        self.age_config = age_config

        # initializing models
        if self.face_detector == 'models/mmod_human_face_detector.dat':
            self.face_detect = dlib.cnn_face_detection_model_v1(self.face_detector)
        elif self.face_detector == 'haarcascade_frontalface_default.xml':
            self.face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + self.face_detector)
        else:
            print(f'model error: does not recognize face detection model {self.face_detector}')
            
        self.age_net = cv2.dnn.readNet(age_config, age_weights)
    
    def detect_face(self, image):
        
        '''
        Face detection with various models
        '''
        if self.face_detector == 'models/mmod_human_face_detector.dat':
            faces = self.face_detect(image, 2)

        elif self.face_detector == 'haarcascade_frontalface_default.xml':
            faces = self.face_detect.detectMultiScale(image, 1.08, 5)

        return faces
    
    def get_rects(self, faces, image):
        
        '''
        gets rectangles from face object, depending on face detection model 
        '''
        rects = []
        if self.face_detector == 'models/mmod_human_face_detector.dat':
            for face in faces:                    
                left = max(face.rect.left(), 0)
                top = max(face.rect.top(), 0)
                right = min(face.rect.right(), image.shape[0]-1)
                bottom = min(face.rect.bottom(), image.shape[1]-1)
            
                rect = [left, top, right, bottom]
                rects.append(rect)

        elif self.face_detector == 'haarcascade_frontalface_default.xml':
            for x,y,w,h in faces:
                rect = [x, y, x+w, y+h]
                rects.append(rect)
        return rects

    def plot_face_rects(self, image, rect):

        '''
        plotting face rectangles on image 
        '''
        left, top, right, bottom = rect

        # Rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        plt.figure(figsize=(12,8))
        # plt.imshow(image)
        # plt.imshow(helper.convert_to_RGB(image))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.waitKey(0)

    def detect_age(self, image, box):

        '''
        Age detection using OpenCV's Deep Neural Network model
        '''

        # setting up model parameters
        model_mean = self.age_model_mean

        # loading model
        age_net = self.age_net

        face = image[box[1]:box[3], box[0]:box[2]]
        
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), model_mean, swapRB = False
        )

        age_net.setInput(blob)
        age_preds = age_net.forward()
        
        return age_preds, box
        
        # except Exception as e:
        #     print(image.shape, box)
        #     print(e)
            
        #     return -1, -1
    
    def plot_age(self, image, age, box):

        '''
        plotting predicted age with face rectangles on image
        '''

        msg = 'Face Detected'

        # plot rectangles around face
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 
                      (00, 200, 200), 2)

        # plot age prediction
        cv2.putText(image, f'{msg}:{age}', (box[0],
                                        box[1] + 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2, cv2.LINE_AA)
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.waitKey(0)
        


    def predict(self, image):

        '''
        make predictions on dataframe
        '''

        image_gray = helper.convert_to_gray(image)

        # start timer for speed analysis
        t_start = time.time()

        # detect face from image
        # print('detect face')
        rects = self.detect_face(image_gray)
        rects = self.get_rects(rects, image)
        # print(rects)
        # print('done')

        t_face = time.time()

        if not rects:
            return -1, -1, -1, -1

        # plot face detection result
        # self.plot_face_rects(image, rect)

        # get first instance of rects
        rect = rects[0]

        age_preds, box = self.detect_age(image, rect)
        age = self.AGES[age_preds[0].argmax()]

        # print('done')

        t_age = time.time()
        # print('age done', t_face-t_start, ages[0])

        if not age or age == -1:
            return -1, -2, -1, -1

        # plot age detection result
        self.plot_age(image, age, box)
    
        return age, box, t_start, t_face, t_age
    
    def test(self, df, output_file):

        BASE_PATH = self.base_path

        results_df = pd.DataFrame()
        count = 0

        for index in range(0, len(df)):

            # get face image path
            image_path = df.iloc[index]['file']
            age = df.iloc[index]['age']
            actual_range = df.iloc[index]['age_range']

            if index%10 == 0:
                results_df.tail(count).to_csv(f'./outputs/{output_file}', mode ='a', header=False)
                if OUTPUT_FLAG:
                    print(results_df.tail(10))
                    print('\n-------------------', 'index:', index, '-------------------\n')
                count = 0


            # load and transform image into correct format
            image = cv2.imread(f'{BASE_PATH}/{image_path}')
            if image.shape[0]*image.shape[1] > 720*640:
                image = helper.resize_image(image)

            image_copy = image.copy()

            predicted_age, rect, t_start, t_face, t_age = self.predict(image_copy)

            # age detection failure
            if predicted_age == -1:
                # print('no ages')
                if OUTPUT_FLAG:
                    if t_start == -1:
                        print(f'face detection error: face for file {image_path} could not be detected, index: {index} ')
                    elif t_start == -2:
                        print(f'age detection error: age for file {image_path} could not be detected, index: {index} ')
                continue

            left, top, right, bottom = rect
            
            # update dataframe
            values = {
                'file': image_path,
                'age': age,
                'age_range': actual_range,
                'rect': f'{left}-{top}-{right}-{bottom}',
                'predicted_range': predicted_age,
                'face_detection_time': t_face - t_start,
                'age_detection_time': t_age - t_face,
                'total_detection_time': t_age - t_start
            }
            
            curr = pd.DataFrame([values])
            results_df = pd.concat([results_df, curr])
            # print(results_df.tail())
            # input('next')

            count += 1

            if index == len(df)-1:
                results_df.tail(count).to_csv(f'./outputs/{output_file}', mode ='a', header=False)
                if OUTPUT_FLAG:
                    print(results_df.tail(count))
                    print('\n-------------------', 'index:', index, '-------------------\n')
                return results_df
        
        return results_df

    def train(self, df):
        
        '''
        trains models to return new parameters
        '''

        # BASE_PATH = self.base_path
        # AGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

        # NUM_OUTPUTS = len(AGES)
        # new_layer = cv2.dnn.blobFromData([], [NUM_OUTPUTS])
        # age_net = self.age_net
        # model_mean = self.age_model_mean

        # age_net.blobFromImage(model_mean)

        # for index in range(0, len(df)):

        #     # get required variables
        #     image_path = df.iloc[index]['file']
        #     actual_range = df.iloc[index]['age_range']
        #     rect = df.iloc[index]['rect']

        #     # load image and convert to proper format
        #     image = cv2.imread(f'{BASE_PATH}/{image_path}')
        #     if image.shape[0]*image.shape[1] > 720*640:
        #         image = helper.resize_image(image)

        #     # training step
        #     last_layer = age_net.getLayer(age_net.getLayerID('last_layer_name'))
        #     age_net.setNetDNNLayer(last_layer.name, new_layer)

        #     box = rect.split('-')
        #     face = image[box[1]:box[3], box[0]:box[2]]

        #     blob = cv2.dnn.blobFromImage(
        #         face, 1.0, (227, 227), model_mean, swapRB = False
        #     )

        #     age_net.setInput(blob)
        #     age_preds = age_net.forward()

        #     age_preds = tf.convert_to_tensor(age_preds[0], dtype=tf.float32)
        #     age_probs = tf.convert_to_tensor(age_probs, dtype=tf.float32)


        #     # calculating loss 
        #     loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #         labels=age_preds, logits=age_probs
        #     ).numpy()
            

        







    


# m = Model(
#     path = '',
#     age_model_mean = (78.4263377603, 87.7689143744, 114.895847746),
#     age_weights = "models/gad/age_deploy.prototxt",
#     age_config = "models/gad/age_net.caffemodel",
#     face_detector='models/haarcascade_frontalface_default.xml'
#     )

# image = cv2.imread('ronaldo.jpg')
# image = helper.resize_image(image)
# image_gray = helper.convert_to_gray(image)

# rects = m.detect_face(image_gray)
# rect = rects[0]

# age_preds, box = m.detect_age(image, rect)
# age = m.AGES[age_preds[0].argmax()]
# m.plot_age(image, age, box)


