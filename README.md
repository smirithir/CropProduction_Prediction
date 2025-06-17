# CropData_Prediction
This project aims to develop a regression model that forecasts crop production (in tons) based on agricultural factors such as area harvested (in hectares), yield (in kg/ha), and the year, for various crops grown in a specific region.

# Agricultural Production Prediction Dashboard

An interactive Streamlit web app to predict crop production (in tons) based on area harvested, yield, region, crop, and year using Machine Learning.

---

## Features

- Cleaned and encoded crop production dataset.
- Exploratory Data Analysis with heatmaps and bar charts.
- Machine learning model (Linear Regression) to predict production.
- Streamlit app for real-time predictions.
- Visual insights into productivity and input-output relationships.

---

## Project Structure

agri-production-predictor/
├── predictive_modeling.py # Data cleaning, modeling, visualization
├── streamlit_app.py # Streamlit app code
├── cleaned_data.csv # Preprocessed dataset
├── production_model.pkl # Trained ML model
├── correlation_heatmap.png # EDA visualization
├── productivity_comparison.png # EDA visualization
├── README.md # Project overview

## Install dependencies
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib

## To run
streamlit crop_data_dashboard.py
