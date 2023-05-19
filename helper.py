import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

# helper functions

def map_age_group(age: int) -> int:

  '''
  maps age value to age group according to model's parameters
  '''

  AGE_RANGES = [(0,2), (4,6), (8,12), (15,20), (25,32), (38,43), (48,53), (60,100)]

  for low, high in AGE_RANGES:
    if age in range(low, high+1):
      return f'({low}-{high})'

  return -1

def create_df(path):
  
  '''
  creates df with image paths and age groups
  '''

  df = pd.DataFrame()

  for age_path in os.listdir(path):
    # print(age_path) 
    age_range = map_age_group(int(age_path))
    if age_range == -1:
      continue

    for image_path in os.listdir(f'{path}/{age_path}'):
      observation = pd.DataFrame([{ # creating observation
          'file': f'{age_path}/{image_path}',
          'age': int(age_path),                         
          'age_range': age_range, 
          }])

      df = pd.concat([df,observation], ignore_index = True)
    
  # df = df.reset_index()
  return df

def random_sampling(df):

  '''
  setting up training and testing dataframes through random sampling
  '''

  train, test = train_test_split(df, test_size = 0.3, shuffle = True, random_state=1234)

  pre_test, post_test = train_test_split(test, test_size = 0.67, shuffle = True, random_state=1234)

  # print(pre_test.shape, post_test.shape)

  return train, pre_test, post_test

def convert_to_RGB(image):

    '''
    converts BGR images to RGB
    '''

    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def convert_to_gray(image):

    '''
    converts BGR images to gray scale
    '''

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_image(image):

    '''
    resizes image to constant dimensions
    '''

    return cv2.resize(image, (720, 640))

def update_observation(df, index, values):

    '''
    adds predictions to dataframe
    '''
    for col, val in values.items():
        df.at[index, col] = val
    
    return df
