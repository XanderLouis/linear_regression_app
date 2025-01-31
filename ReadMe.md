# Power Output Prediction
Power Output Prediction using Sklearn, FastAPI, and a Streamlit app

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
  - [1. Train and Save the Model](#1-train-and-save-the-model)
  - [2. Deploy FastAPI](#2-deploy-fastapi)
  - [3. Run Streamlit](#3-run-streamlit)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Example Input and Output](#example-input-and-output)
- [File Structure](#file-structure)
- [License](#license)

## Description
This project provides an API and a Streamlit application for predicting power output (PE) based on environmental factors. The model uses **Linear Regression** from Scikit-Learn, trained on features including:

- Ambient Temperature (AT)
- Exhaust Vacuum (V)
- Ambient Pressure (AP)
- Relative Humidity (RH)

The API is deployed using **FastAPI**, and a **Streamlit app** provides an interactive interface for users to input values and receive predictions.

## Requirements
To set up and run this project properly, you'll need the following Python packages:

- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pandas`
- `joblib`
- `numpy`
- `streamlit`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
