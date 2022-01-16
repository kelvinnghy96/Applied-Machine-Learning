# Table of Content


# Installation Guide
1. Click [here](https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/archive/refs/heads/main.zip) to download ```Air Pressure System Failure for Scania Trucks``` ZIP file.
2. Extract downloaded ZIP file.
3. Run the source code in the ```code``` folder to see the result.


# ABSTRACT
Stroke, a disease which impact on arteries connecting to brain and occur when a blood vessel is blocked from transferring nutrients and oxygen to the brain. It is the third major leading causes of death in Malaysia, and it reach 9.80% of Malaysia’s total deaths in year 2018. With early detection on stroke disease, various of preventive action can be took to reduce the damage dealt to the stroke patient, therefore, numerous of research is done all over the world to predict stroke with data. In this assignment, 4 machine learning model approach which are Naïve Bayes, Logistic Regression, Random Forest, and Support Vector Machine (SVM) have been built for stroke prediction. Here, One-hot encoding has been used to convert categorical data into binary variable to provide more detail information in model training, synthetic minority over-sampling technique (SMOTE) has been used for class balancing in train dataset and min-max normalization technique is used to convert all numeric value into common scale. After analyzing and comparing between the 4-model built, the optimum stroke prediction model among the 4 model is 93.59% accuracy.

# INTRODUCTION, RESEARCH GOAL & OBJECTIVES
Data science has become a trend in year 2021. A lot of industry including healthcare sector using data to improve the production rate and efficiency in their sector. Healthcare sector is one of the important sectors that used high accuracy prediction model to take preventive action on various disease and to reduce disease mortality rate of a country. Stroke disease has been selected in this assignment because it is one of the major diseases causes death over the world and the most important reason is stroke can be prevented and cure if it able to be predict and detected at early stage and therefore model accuracy is very important for a disease prediction model as early detection or prediction can lead to earlier preventive action and save a life.

The objective and research goal of this assignment is to train 4 machine learning model which are Naïve Bayes, Logistic Regression, Random Forest, and SVM then compare among the model to retrieve a stroke prediction model with the highest accuracy. In this assignment 12 clinical features including patient demography, body mass index, average glucose level, high blood pressure status and heart disease status are collected and used to train the model. Data are collected from more than 5000 individuals while all the sensitive data has been masked and removed. These datasets are integrated and transformed to remove abnormal and missing value while multiple data pre-process technique such as one-hot encoding, min-max normalization and class balancing has been applied on the dataset to train a better stroke prediction model.

# MODEL USED
- Naïve Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)


# DATASET DESCRIPTION
This assignment used a stroke prediction dataset which contain 5110 observations with 13 attributes. This dataset contains patient id to ensure the uniqueness of each observation and 12 other attributes which will be explained in detail. Gender, categorical data with categories of male and female. Age, continuous data with normal range between 0 to 120. Age categories, categorical data with categories of infants, adults, children, older adults, and teens. Smoking status, categorical data with categories of formerly, never and smokes. Married status, categorical data with categories of yes and no which determine whether an individual is married previously. Employment status with categories of children, government job, never worked, private and self-employed. Region type, categorical data with categories of rural and urban. Body mass index, continuous data with normal range between 20 to 60. Average glucose level, continuous data with normal range between 60 to 240. High blood pressure, categorical data with categories of 0 and 1 which determine whether an individual is having high blood pressure. Heart disease, categorical data with categories of 0 and 1 which determine whether an individual is having a heart disease. Stroke status, target variable in this dataset and categorical data with categories of yes and no which determine whether an individual is having stroke
