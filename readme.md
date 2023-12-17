# Diabetes Prediction Jupyter Notebook

## Overview

This Jupyter notebook explores the task of predicting diabetes based on a comprehensive dataset. The analysis includes data preprocessing, exploratory data analysis, feature engineering, and the implementation of various machine learning models. The goal is to identify an effective model for predicting diabetes and understand the factors influencing the predictions.

## Table of Contents

1. [Libraries Import](#libraries-import)
2. [Data Read-In](#data-read-in)
3. [Column Preprocessing](#column-preprocessing)
4. [Selecting Binary Columns](#selecting-binary-columns)
5. [Features Proportion for Diabetes and Non-Diabetes Records](#features-proportion)
6. [Continuous Variables Distribution](#continuous-variables-distribution)
7. [Binning Variables](#binning-variables)
8. [Confusion Matrix Metrics Function](#confusion-matrix-metrics)
9. [Preprocessing for Machine Learning](#preprocessing-for-ml)
10. [Logistic Regression with Recursive Feature Elimination (RFE)](#logistic-regression-rfe)
11. [Gradient Boosting Classifier Model](#gradient-boosting-model)
12. [K-Nearest Neighbors (KNN) Model](#knn-model)
13. [Naive Bayes Model](#naive-bayes-model)
14. [SMOTEENNing Data](#smoteenn)
15. [SMOTEENNed Naive Bayes](#smoteenned-naive-bayes)
16. [Acknowledgment](#acknowledgment)
17. [Conclusion](#conclusion)

## Instructions

- The notebook is organized sequentially, and each section is marked with a heading for clarity.
- Ensure all necessary libraries are installed before running the notebook. You can install them using `pip install -r requirements.txt`.
- The dataset should be placed in the same directory as the notebook and named "data.csv."
- Follow the step-by-step analysis to understand the data, preprocessing, and the performance of different machine learning models.
- Adjust parameters, features, or models as needed for further experimentation.

## Acknowledgment

Special thanks to the [julnazz](https://www.kaggle.com/julnazz) author of the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/julnazz/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2021.csv) on Kaggle.

## Conclusion

This notebook provides a comprehensive analysis of predicting diabetes using various machine learning models. It emphasizes the trade-offs between different models, including logistic regression, gradient boosting, k-nearest neighbors, and naive Bayes. The impact of preprocessing techniques like feature engineering and SMOTEENN resampling is also explored.

The findings indicate that while certain models achieve high recall, they may sacrifice accuracy and precision.


