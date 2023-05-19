# ROBOCUP

# Introduction

This project was made for Wesleyan's QAC239 Proseminar: Machine Learning Methods for Audio and Video Analysis class. 

# Directory Tree

    QAC239_FINALPROJECT (root)
    ├── models: various ML models used in project
      └── gad: OpenCV age classication model
      └── gad.zip
      └── haarcascade_frontalface_default.xml: Haar Cascade facial detection model
      └── mmod_human_face_detector.dat: MMOD DeepNN facial detection model
      └── mmod_human_face_detector.dat.bz2
    ├── outputs: robocup server
      └── plots: graph plots
      └── predictions: prediction csv files
      └── sample images: sample testing images and prediction outputs 
    ├── proposal: project proposal
    ├── venv: virtual environment
    ├── .gitignore
    ├── evaluation.py: model evaluation file (accuracy and speed)
    ├── helper.py: helper functions
    ├── LICENSE: basic MIT license
    ├── live_detect.py: real time age classifier
    ├── main.py: main file
    ├── model.py: model class
    ├── plot.py: plots file
    ├── README.md
    └── requirements.txt: required python packages

# Virtual Environment Set-up

## Creating virtual environment

```bash
python -m venv env_name
```

## Activating virtual environment
Windows: 
```bash
.\env_name\scripts\activate
```

## installing required packages
```bash
pip install -r requirements.txt
```

# Running the project

## set-up

before running main.py, you will need to download the facial-ages dataset into this directory and rename it to assets. You can find the dataset set here: 

  https://www.kaggle.com/datasets/frabbisw/facial-age

## main.py

Running main.py will take you through all the steps I had to set up the dataframe, test the model and train:

```bash
python main.py
```

## live_detect.py

Running live_detect.py will start up the real time age classfier, using your device's web camera
```bash
python live_detect.py
```

# References

    https://talhassner.github.io/home/publication/2015_CVPR
    https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf
    https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender
    https://www.kaggle.com/datasets/frabbisw/facial-age
