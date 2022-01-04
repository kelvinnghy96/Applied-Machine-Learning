# Import library
library(psych)
library(DataExplorer)
library(dplyr)
library(ggplot2)
library(caTools)
library(mltools)
library(data.table)
library(reshape2)
library(lattice)
library(grid)
library(UBL)
library(e1071)
library(caret)
library(randomForest)
library(tictoc)

# Set seed to 2021
set.seed(2021)

# Get working directory
work_dir = getwd()
work_dir

# Read dataset from .csv file
data = read.csv(paste(work_dir,"stroke-prediction-data.csv",sep="/") , header = T)

# View dataset
head(data)
dim(data)
str(data)
summary(data)
describe(data)

# Dataset description
'
V1 - patient_id - (id)
V2 - gender
V3 - age
V4 - age_categories
V5 - smoking_status
V6 - married_status
V7 - employment_status
V8 - region_type
V9 - body_mass_index
V10 - avg_glucose_level
V11 - high_blood_pressure
V12 - heart_disease
V13 - stroke_status - (target variable)
'

# body_mass_index data type is 'character' due to 'N/A' text which is incorrect
# use as.numeric will automatically convert non numeric value to N/A value
class(data$body_mass_index)
data$body_mass_index = as.numeric(data$body_mass_index)
class(data$body_mass_index)

# Check data for categorical column
table(data$gender)
table(data$age_categories)
table(data$smoking_status)
table(data$employment_status)
table(data$region_type)
table(data$heart_disease)
table(data$stroke_status)

# Drop ID column
df <- select(data, -patient_id )

# Remove 'Other' category from gender column
df = df[(df$gender == 'Male' | df$gender == 'Female'),]
table(df$gender)

# Convert empty string and unknown value to NULL
df$smoking_status[df$smoking_status=="Unknown"] = NA
df[df==""] = NA

# Check and remove missing data in the dataset
colSums(sapply(df,is.na))
plot_missing(df,group = list("Good" = 0, "Ok" = 0.05, "Bad" = 0.3, "Remove" = 1))
df = na.omit(df)
colSums(sapply(df,is.na))

# Check for outlier and abnormal data
describe(df)

# Remove body_mass_index with record more than 100 which is out of normal range
df = df[!(df$body_mass_index > 100),]

# Remove avg_glucose_level with record more than 300 which is out of normal range
df = df[!(df$avg_glucose_level > 300),]

# Remove negative value in age and body_mass_index
df = df[!( df$age < 0 | df$body_mass_index < 0),]
describe(df)

# Check and remove value for age under 14 but employment status are not 'Children'
subset(df,df$age < 14 & df$employment_status!="Children",select=c(age,employment_status))
df = df[!( df$age < 14 & df$employment_status!="Children"),]

# Find correlation between all attributes except target variable
corr_Matrix <- round(cor(df[sapply(df,is.numeric)]), 2)
print(corr_Matrix)

# Build correlation matrix heatmap
cormat_heatmap <- melt(corr_Matrix)
head(cormat_heatmap)
ggheatmap = ggplot(data = cormat_heatmap, aes(x=Var1, y=Var2, fill=value)) + geom_tile()
ggheatmap + geom_text(aes(Var2, Var1, label = value), color = "white", size = 4)


# One-hot encoding categorical data with data table
col_names <- c('gender', 'age_categories' ,'smoking_status','married_status',
               'employment_status','region_type')
df[,col_names] <- lapply(df[,col_names] , factor)
df2 <- one_hot(as.data.table(df))

# Change back to data frame type
df2 = as.data.frame(df2)

# Labeling the target variable values
# 'No' to 0 while 'Yes' to 1
df2$stroke_status <- factor(df2$stroke_status, levels=c("No","Yes"), 
                            labels=c("0", "1"))

# Checking the class distribution of original data
table(df2$stroke_status)
class(df2)
prop.table(table(df2$stroke_status)) 
barplot(prop.table(table(df2$stroke_status)), col = rainbow(3), 
        ylim = c(0, 1), main = "Class Distribution",xlab = "Stoke Status",
        ylab = "Percentage")


# Split data into train and test dataset
ind <- sample(2, nrow(df2), replace=TRUE, prob=c(0.7, 0.3))
train <- df2[ind==1,]
test <- df2[ind==2,]
dim(train)
dim(test)

# Class balancing on train dataset
table(train$stroke_status)
train_bal = SmoteClassif(train$stroke_status ~ ., train, C.perc = "balance")
table(train_bal$stroke_status)
prop.table(table(train_bal$stroke_status)) 
barplot(prop.table(table(train_bal$stroke_status)), col = rainbow(3), 
        ylim = c(0, 1), main = "Class Distribution",xlab = "Stoke Status",
        ylab = "Percentage")

# Min- Max Normalization for numerical value 
# One-hot encoded column will remain 0 and 1
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Min- Max Normalization train data
train_bal_norm_tmp <-as.data.frame(lapply(select(train_bal, -stroke_status), normalize))
head(train_bal_norm_tmp)
train_bal_norm <- cbind(train_bal_norm_tmp,stroke_status= train_bal$stroke_status)
head(train_bal_norm)
str(train_bal_norm)

# Min- Max Normalization test data
test_norm_tmp <-as.data.frame(lapply(select(test, -stroke_status), normalize))
head(test_norm_tmp)
test_norm <- cbind(test_norm_tmp,stroke_status= test$stroke_status)
head(test_norm)
str(test_norm)

# Preprocessed train and test dataset
train_final = train_bal_norm
test_final = test_norm

# Separate target variable for train and test dataset
xtrain = select(train_final, -stroke_status)
ytrain = train_final$stroke_status
xtest = select(test_final, -stroke_status)
ytest = test_final$stroke_status


# Naive Bayes
# Naive Bayes model tuning using grid search method
# 936.51 sec elapsed
search_grid = expand.grid(usekernel = c(TRUE, FALSE),fL = 0:5,
                          adjust = seq(0, 5, by = 1))
tic()
nb_tune_model = train(xtrain, ytrain, 'nb', metric="Accuracy",tuneGrid = search_grid)
toc()

# The final values used for the model were fL = 0, usekernel = TRUE and adjust = 1
print(nb_tune_model)
plot(nb_tune_model)
attributes(nb_tune_model)

# Building the optimum NB model using default value,fL=0,usekernel=TRUE,adjust=1 
# 39.19 sec elapsed
tic()
nb_model = train(xtrain, ytrain, 'nb', metric="Accuracy")
toc()
nb_model

# Plot Variable performance
varImp(nb_model)
plot(varImp(nb_model))

# Pick only attribute with importance above 0.3
nbcol_name = c("age", "age_categories_OlderAdults", "age_categories_Adults","high_blood_pressure","heart_disease","avg_glucose_level")
nbxtrain = xtrain[,nbcol_name]

# Building the optimum NB model with Feature Selection
# 14.22 sec elapsed
tic()
nb_model2 = train(nbxtrain, ytrain, 'nb', metric="Accuracy")
toc()
nb_model2

# Predict testing set from "nb_model2"
nb_model_predict <- predict(nb_model2, xtest)

# Naive Bayes model accuracy
nb_cm = confusionMatrix(nb_model_predict, ytest)
nb_acc = nb_cm[["overall"]][["Accuracy"]]
nb_acc


# Logistic Regression
# Logistic Regression model training
logreg_model = glm(stroke_status ~., train_final, family = binomial)
summary(logreg_model)

# Select attributes with 2* and above
lgcol_name = c("age", "smoking_status_Formerly", "smoking_status_Never","high_blood_pressure", "heart_disease","avg_glucose_level","stroke_status")
lgxtrain = train_final[,lgcol_name]

# Logistic Regression Feature Selection model training
logreg_model2 = glm(stroke_status ~., lgxtrain, family = binomial)
summary(logreg_model2)

# Predicting the Test set results
logreg_model_predict_tmp = predict(logreg_model2, type = 'response', xtest )
logreg_model_predict = ifelse(logreg_model_predict_tmp > 0.5, 1, 0)
logreg_cm = table(ytest, logreg_model_predict)
logreg_cm

# Logistic Regression model accuracy
logreg_acc = sum(diag(logreg_cm))/sum(logreg_cm)
logreg_acc


# Random Forest
# Random Forest model tuning using grid search method
# 290.53 sec elapsed
tunegrid <- expand.grid(.mtry=c(1:8))
tic()
rf_tune_model <- train(stroke_status~.,data = train_final, method="rf", metric="Accuracy", tuneGrid=tunegrid)
toc()

# The final value used for the model was mtry = 6
print(rf_tune_model)
plot(rf_tune_model)
attributes(rf_tune_model)

# Building the optimum RF model using mtry = 6
# 6.05 sec elapsed
tic()
rf_model <- randomForest(stroke_status~.,data = train_final,
                    mtry=6,
                    importance = TRUE,
                    proximity = TRUE)
toc()
print(rf_model)
attributes(rf_model)

# RF model accuracy
rf_model_predict <- predict(rf_model, test_final)
rf_cm <- table(rf_model_predict, ytest)
rf_cm
rf_acc = sum(diag(rf_cm)/sum(rf_cm))
rf_acc


# SVM
# SVM model tuning using grid search method
# 168.57 sec elapsed
tic()
svm_tune_model = tune(svm, stroke_status~., data=train_final,
                   ranges = list(epsilon = seq (0, 1, 0.1), cost = 2^(0:2)))
toc()
plot (svm_tune_model)
summary (svm_tune_model)

# SVM optimum model, epsilon = 0, cost = 4, kernel = radial
svm_opt_model = svm_tune_model$best.model
summary(svm_opt_model)

# Building the optimum SVM model
svm_best_model <- svm(stroke_status~., data = train_final, epsilon = 0, 
                      cost = 4, kernel = c("radial"))
summary(svm_best_model)

# SVM model accuracy
svm_model_predict = predict(svm_best_model, xtest)
svm_model_predict
svm_cm = table(Predicted = svm_model_predict, Actual = ytest)
svm_cm
svm_acc = sum(diag(svm_cm))/sum(svm_cm)
svm_acc

# Accuracy of all 4 models
nb_acc
logreg_acc
rf_acc
svm_acc

