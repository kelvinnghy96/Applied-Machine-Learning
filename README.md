# Stroke Prediction with Data Science

## Table of Content
- [Stroke Prediction with Data Science](#stroke-prediction-with-data-science)
  - [Table of Content](#table-of-content)
  - [Installation Guide](#installation-guide)
  - [Abstract](#abstract)
  - [1.1 Introduction](#11-introduction)
  - [1.2 Objectives & Research Goal](#12-objectives--research-goal)
  - [1.3 Model Used](#13-model-used)
  - [1.4 Dataset Description](#14-dataset-description)
  - [1.5 RELATED WORKS](#15-related-works)
  - [1.6 Learning Technique and Library Used](#16-learning-technique-and-library-used)
  - [1.7 DATASET PREPARATION](#17-dataset-preparation)
    - [1.7.1 Data Cleansing](#171-data-cleansing)
    - [1.7.2 Correlation Matrix](#172-correlation-matrix)
    - [1.7.3 One-hot Encoding](#173-one-hot-encoding)
    - [1.7.4 Class Balancing](#174-class-balancing)
    - [1.7.4 Min-max Normalization](#174-min-max-normalization)
  - [1.8 Naïve Bayes](#18-naïve-bayes)
  - [1.9 Logistic Regression](#19-logistic-regression)
  - [1.10 Random Forest](#110-random-forest)
  - [1.11 Support Vector Machine (SVM)](#111-support-vector-machine-svm)
  - [1.12 Analysis & Recommendation](#112-analysis--recommendation)
  - [License](#license)

## Installation Guide
1. Click [here](https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/archive/refs/heads/main.zip) to download ```Stroke Prediction with Data Science``` ZIP file.
2. Extract downloaded ZIP file.
3. Run the source code in the ```code``` folder to see the result.


## Abstract
Stroke, a disease which impact on arteries connecting to brain and occur when a blood vessel is blocked from transferring nutrients and oxygen to the brain. It is the third major leading causes of death in Malaysia, and it reach 9.80% of Malaysia’s total deaths in year 2018. With early detection on stroke disease, various of preventive action can be took to reduce the damage dealt to the stroke patient, therefore, numerous of research is done all over the world to predict stroke with data. In this assignment, 4 machine learning model approach which are Naïve Bayes, Logistic Regression, Random Forest, and Support Vector Machine (SVM) have been built for stroke prediction. Here, One-hot encoding has been used to convert categorical data into binary variable to provide more detail information in model training, synthetic minority over-sampling technique (SMOTE) has been used for class balancing in train dataset and min-max normalization technique is used to convert all numeric value into common scale. After analyzing and comparing between the 4-model built, the optimum stroke prediction model among the 4 model is 93.59% accuracy.

## 1.1 Introduction
Data science has become a trend in year 2021. A lot of industry including healthcare sector using data to improve the production rate and efficiency in their sector. Healthcare sector is one of the important sectors that used high accuracy prediction model to take preventive action on various disease and to reduce disease mortality rate of a country. Stroke disease has been selected in this assignment because it is one of the major diseases causes death over the world and the most important reason is stroke can be prevented and cure if it able to be predict and detected at early stage and therefore model accuracy is very important for a disease prediction model as early detection or prediction can lead to earlier preventive action and save a life.

## 1.2 Objectives & Research Goal
The objective and research goal of this assignment is to train 4 machine learning model which are Naïve Bayes, Logistic Regression, Random Forest, and SVM then compare among the model to retrieve a stroke prediction model with the highest accuracy. In this assignment 12 clinical features including patient demography, body mass index, average glucose level, high blood pressure status and heart disease status are collected and used to train the model. Data are collected from more than 5000 individuals while all the sensitive data has been masked and removed. These datasets are integrated and transformed to remove abnormal and missing value while multiple data pre-process technique such as one-hot encoding, min-max normalization and class balancing has been applied on the dataset to train a better stroke prediction model.

## 1.3 Model Used
- Naïve Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## 1.4 Dataset Description
This assignment used a stroke prediction dataset which contain 5110 observations with 13 attributes. This dataset contains patient id to ensure the uniqueness of each observation and 12 other attributes which will be explained in detail. Gender, categorical data with categories of male and female. Age, continuous data with normal range between 0 to 120. Age categories, categorical data with categories of infants, adults, children, older adults, and teens. Smoking status, categorical data with categories of formerly, never and smokes. Married status, categorical data with categories of yes and no which determine whether an individual is married previously. Employment status with categories of children, government job, never worked, private and self-employed. Region type, categorical data with categories of rural and urban. Body mass index, continuous data with normal range between 20 to 60. Average glucose level, continuous data with normal range between 60 to 240. High blood pressure, categorical data with categories of 0 and 1 which determine whether an individual is having high blood pressure. Heart disease, categorical data with categories of 0 and 1 which determine whether an individual is having a heart disease. Stroke status, target variable in this dataset and categorical data with categories of yes and no which determine whether an individual is having stroke

## 1.5 RELATED WORKS
While searching for stroke prediction model relevant previous work, search result shows the number of stroke prediction model is increasing year by year. Various of model are used in building stroke prediction model but the dataset used in each related work are relatively small to fully explore the potential of machine learning model and shorten the duration to train the model. This assignment also used relatively small stroke prediction dataset to build and train the model.

For handling null value or empty string, most of the relevant work will handle null value with simple method such as single imputation or complete case analysis which is also used in this assignment. Advantage of using complete case analysis is it will remove noise variable and retain only useful and accurate data, but it will also further reduce the size of the dataset and affect the efficiency of model training process.

Most used machine learning model in each relevant work were Random Forest, SVM, Artificial Neural Network and Decision Tree. It’s hard to compare and decide best performance model in each relevant work as the data characteristic used is different in each relevant work but SVM perform better in most of the studies follow up with Random Forest and Artificial Neural Network. Random Forest and SVM are also used in this assignment but the model performance result shows that Random Forest perform better than SVM in this assignment.

Model validation is important in model building so that the model can have certain accuracy when dealing with real world industry data. In most of the relevant work, common internal validation methods are used such as train test data split and cross validation. Train test data split validation is used in this assignment with a ratio of 0.7 because cross validation with multiple folds will require more training time and consume more local machine resource which will slow down the process of this assignment, therefore, split validation that faster speed in training model is used in this assignment.

Performance metric is important to determine which model perform better. In most of the relevant work, F-score is used when dealing with imbalance data while accuracy is used when dealing with balance data. In this assignment, accuracy will be the main performance metric to determine the performance of each model built. 

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/table1.png" /></a>
</p>

Summary table is created above based on review on previous relevant work. Every process built and model trained in this assignment will be done based on the reference on the above summary table.

## 1.6 Learning Technique and Library Used
The learning technique used in this assignment is supervised learning technique and the package and library used is listed in figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure1.png" /></a>
</p>

The machine learning method use in this assignment are Naïve Bayes, Logistic Regression, Random Forest, and SVM as these methods are good classifier to deal with categorical target variable. The metric used in this assignment to evaluate the performance of the model is ```accuracy```.

## 1.7 DATASET PREPARATION
### 1.7.1 Data Cleansing
Before start with dataset preparation, seed is set at ```2021``` to ensure getting back same result each time the process is rerun. When checking dataset using ```str(data)```, wrong data type is detected on ```body_mass_index``` which is ```character``` type instead of ```numeric``` type as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure2.png" /></a>
</p>

The ```body_mass_index``` data type is ```character``` due to ```N/A``` text value in the column, therefore, ```as.numeric()``` function is used to automatically convert ```N/A``` text value to null value and convert the other data to ```numeric``` data type as figures below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure3.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure4.png" /></a>
</p>

Patient id is dropped from the dataset and an extra category of ```Other``` is removed from the Gender column as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure5.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure5.1.png" /></a>
</p>

Both missing data and empty string is detected in ```avg_glucose_level```,
```employment_status```, ```married_status```, ```heart_disease```, ```high_blood_pressure```, ```body_mass_index```, ```smoking_status``` and removed from the dataset as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure6.png" /></a>
</p>

Outlier value and abnormal value is detected and removed from ```body_mass_index```, ```avg_glucose_level``` and ```age``` as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure7.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure8.png" /></a>
</p>

Children under age ```14``` in Malaysia are not allowed to be employed, therefore, data are checked for age under ```14``` and ```employment_status``` are not ```Children``` as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure9.png" /></a>
</p>

### 1.7.2 Correlation Matrix
Correlation matrix heatmap is built to determine the correlation between all numeric attributes as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure10.png" /></a>
</p>

From the figure above, there are no correlation between attribute that is more than ```0.8```, therefore, no attribute is removed from dataset at this step.

### 1.7.3 One-hot Encoding
```One-hot encoding``` technique is used to convert ```categorical``` variable into ```binary``` variable so that the variables can provide more detail information for model training. ```Categorical``` column need to be convert to ```factor``` type only able to apply with ```one-hot()``` function as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure11.png" /></a>
</p>

```Stroke_status``` which is the ```target variable``` is labelled for stroke prediction in later phase as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure12.png" /></a>
</p>

Dataset is split into ```train``` and ```test``` data with a ratio of ```7:3``` respectively as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure13.png" /></a>
</p>

### 1.7.4 Class Balancing
Once the dataset is split into train and test data, ```class balancing``` is performed on the ```train data``` only. The reason class balancing is performed only on train data is because train data need balanced class to train model while test data need to remain as similar as real world data to get a correct accuracy.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure14.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure15.png" /></a>
</p>

After perform class balancing using SMOTE technique to oversample the dataset, the distribution of target variable is balanced.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure16.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure17.png" /></a>
</p>

### 1.7.4 Min-max Normalization
Min-max normalization is performed on both train and test data as preprocess modelling need to be exactly same for train data, test data as well as real world data when the model is in production.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure18.png" /></a>
</p>

## 1.8 Naïve Bayes
The first model build is Naïve Bayes model. The hyperparameter of Naïve Bayes model is tuned with ```grid search``` method as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure19.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure20.png" /></a>
</p>

```3``` hyperparameter in Naïve Bayes were tuned which are ```fL```, ```usekernel``` and ```adjust```. 
```fL``` hyperparameter allow user to include Laplace smoother, ```usekernel``` hyperparameter allows user to use a kernel density estimate for continuous variables against a guassian density estimate while ```adjust``` hyperparameter is referring to the bandwidth of kernel density. 
The final value used to build Naïve Bayes model after tuned were ```fL = 0```, ```usekernel = TRUE``` and ```adjust = 1```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure21.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure22.png" /></a>
</p>

From the above figure, it’s clearly shown that ```age``` attribute plays an importance role in model training. The higher the age, the older the age categories, the more importance the role in Naïve Bayes stroke prediction model building. Naïve Bayes stroke prediction model are train again under feature selection with only attributes with importance above ```0.3```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure23.png" /></a>
</p>

The above figure is the ```confusion matrix``` of Naïve Bayes model, and the ```precision``` of Naïve Bayes model can be calculated which is ```35 / (35 + 15) = 0.7``` while the ```recall``` of Naïve Bayes model is ```35 / (35 + 228) = 0.13```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure24.png" /></a>
</p>

The accuracy of Naïve Bayes stroke prediction model is ```76.03%```.

## 1.9 Logistic Regression
The next model build is Logistic Regression model. The Logistic Regression model is train as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure25.png" /></a>
</p>

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure26.png" /></a>
</p>

```Feature selection``` is performed in Logistic Regression model based on significant attributes with ```2 *``` and above based on above figure.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure27.png" /></a>
</p>

After ```feature selection``` is performed only ```7``` attributes are retained in Logistic Regression as figure above.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure28.png" /></a>
</p>

The above figure is the ```confusion matrix``` of Logistic Regression model, and the ```precision``` of Logistic Regression model can be calculated which is ```39 / (39 + 270) = 0.13``` while the ```recall``` of Logistic Regression model is ```39 / (11 + 39) = 0.78```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure29.png" /></a>
</p>

The accuracy of Logistic Regression stroke prediction model is ```72.28%```.

## 1.10 Random Forest
The third model build is Random Forest model. The Random Forest model is tuned using ```grid search``` method as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure30.png" /></a>
</p>

Only one hyperparameter in Random Forest is tuned which is the ```mtry``` hyperparameter as tuning on both ```mtry``` and ```ntree``` parameter will consume a lot of resource of the local machine and the duration took few hours, therefore, in this assignment only ```mtry``` hyperparameter is
tuned for Random Forest. 
```Mtry``` hyperparameter will randomly sample variables based on the number assigned as candidate at each split. The final value of mtry hyperparameter tune from ```grid search``` method is ```mtry = 6```, therefore, Random Forest model are train with hyperparameter ```mtry = 6```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure31.png" /></a>
</p>

The above figure is the ```confusion matrix``` of Random Forest model, and the ```precision``` of Random Forest model can be calculated which is ```4 / (46 + 4) = 0.08``` while the ```recall``` of Random Forest model is ```4 / (19 + 4) = 0.17```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure32.png" /></a>
</p>

The ```accuracy``` of Random Forest stroke prediction model is ```93.59%```.

## 1.11 Support Vector Machine (SVM)
The next model build is SVM model. The SVM model is tuned using ```grid search``` method as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure33.png" /></a>
</p>

```3``` hyperparameter in SVM were tuned which are ```epsilon```, ```cost```, and ```kernel```. 
```Epsilon``` hyperparameter represent the margin that user allow and tolerate that penalty is not given to error, ```cost``` hyperparameter also referring to cost of misclassification where user decide how much data that SVM are allowed to misclassify while ```kernel``` hyperparameter is referring to the method of mathematical function used to deal with input data and transform into. 
The final value used to build SVM model after tuned ```epsilon = 0```, ```cost = 4```, ```kernel = radial```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure34.png" /></a>
</p>

The above figure is the ```confusion matrix``` of SVM model, and the ```precision``` of SVM model can be calculated which is ```7 / (7 + 43) = 0.14``` while the ```recall``` of SVM model is ```7 / (7 + 54) = 0.11```.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure35.png" /></a>
</p>

The ```accuracy``` of SVM stroke prediction model is ```90.43%```.

## 1.12 Analysis & Recommendation
Total of 4 models have been trained in this assignment which are Naïve Bayes, Logistic Regression, Random Forest and SVM. As small dataset gets overfitted easily, therefore, these 4 models is chosen because they can perform better when dealing with small dataset. All four model has been trained and tested with preprocessed data and preprocess model and the main performance metric which is accuracy for the 4 model are recorded as figure below.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/figure36.png" /></a>
</p>

Based ```accuracy``` of 4 trained model in this assignment, ```Random Forest``` model will be the ```champion model``` in this assignment with the highest accuracy.

<p align="center">
  <a href=##><img src="https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/public/image/table2.png" /></a>
</p>

Naïve Bayes model and Logistic Regression model have a high precision and recall respectively compared to Random Forest model and SVM model which mean Naïve Bayes model can get the true positive value and manage to determine patient who are having stroke while Logistic Regression model manage to get a high amount of false negative which mean logistic regression can’t determine the patient who having stroke but it somehow predicted those who are not having stroke might have stroke in the future which preventive action can be take in advance.

Although Random Forest model and SVM model don’t have high precision rate and recall rate, but their accuracy is higher compared to Naïve Bayes model and Logistic Regression model which are 93.59% and 90.43% respectively.

SVM outperform Logistic Regression because SVM finds the optimum distance between line and support vectors to separate class and this lower risk of classification error, while logistics regression can have different decision line with various weight that close to optimal point.

Random Forest outperformed Logistic Regression because the explanatory variable in this dataset is more while Logistic Regression can only perform better when the explanatory variable is more than noise variable.

SVM outperform Naïve Bayes because SVM deal with the interaction between attribute until certain level, but Naïve Bayes treat all attributes independently which cant interpret a deeper level of relationship between attributes but Naïve Bayes train faster than SVM as only probability of each class needed to be calculated in Naïve Bayes.

Random Forest outperformed Naïve Bayes because it’s have a much more complex and large model size compare to Naïve Bayes while Naïve Bayes is simple model and cannot cater complicated data behavior, therefore, Random Forest have better performance than Naïve Bayes with a dataset with complex behavior but advantage of Naïve Bayes is that it can quickly adapt to the changes in any new dataset while Random Forest need to rebuild whenever there’s changes in dataset else it will lead to overfitting.

In most of the related work, SVM and Random Forest are used more in stroke prediction model building compared to Naïve Bayes and Logistic Regression as in the real world industry, stroke is a critical disease that can ruin a patient life, therefore, a model with higher accuracy will be prioritize in the real world situation and same case in this assignment where Random Forest stroke prediction model will be recommended as it is the champion model and it has the highest accuracy compare to the other model.

## License
Click [here](https://github.com/kelvinnghy96/Stroke-Prediction-with-Data-Science/blob/main/LICENSE) to view license.