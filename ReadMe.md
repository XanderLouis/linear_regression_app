# Power Output Prediction
Power Output Prediction using Sklearn, FastAPI and streamlit app

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [1. Train and Save the Model](#1-train-and -save-the-model)
  - [2. Deploy FastAPI](#2-deploy-fastapi)
  - [3. Run Streamlit](#3-streamlit)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Example Input and Output](#example-input-and-output)
- [File Structure](#file-structure)
- [License](#License)

## Description
This project provides and API and a Streamlit application for predicting power output (PE) based on environmental factors. The model uses Linear Regression Scikit-Learn, trained on featuring inluding:

- Ambient Temperature (AT)
- Exhaust Vacuum (V)
- Ambient Pressure (AP)
- Relative Humidity (RH)

The API is deployed using FastAPI, and a Streamlit app provides an interactive interface for users to input values and get needed predictions.

#Requirements
to set up and run this project, you'll need the following python

- 'fastapi'
- 'uvicorn'
- 'scikit-learn'
- 'pandas'
- 'joblib'
- 'numpy'
- 'streamlit'

You can install these dependencies by running:
'''bash
pip install -r requirements.txt
'''

## Getting started
follow these steps to set up and run the project.

1. Train and Save Model

  Train a Linear Regression Model using scikit-learn, and save the trained model to a file for deployment
  '''bash
  python linear_regression_model.py
  '''

2. De