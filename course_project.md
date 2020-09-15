---
title: 'How well of a particular activity do you do?'
output:
  html_document:
    keep_md: yes
  pdf_document: default
  word_document: default
---

Author: Armin Najarpour Foroushani

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Goal

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Any of the other variables may be used to predict with.

### Analysis and results

#### Preparing and cleaning data

Load related packages:
First we load training and testing data


Then we load training and testing data

```r
training <- read.csv("pml-training.csv", header=TRUE)
testing <- read.csv("pml-testing.csv", header=TRUE)
```
We rename the "problem_id" column in the testing data to "classe" in order to use the same name in both sets. 

```r
colnames(testing)[160] <- "classe"
```
Next, we select those columns that contain data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

```r
testing <- testing %>% select(user_name, 8:160)
training <- training %>% select(user_name, 8:160)
```
Many columns in the test set are mostly NA values or contain no numerical values. So, prediction based on these data will be meaningless. Even imputation may not yield a meaningful prediction. We remove these columns from our analysis:

```r
non_nan_col <- colSums(is.na(testing)) == 0
testing <- testing[, non_nan_col]
training <- training[, non_nan_col]
```
So, 53 variables (features) were left.

#### Exploratory data analysis

To explore the data, we selected gyros forearm predictors and visualized them using feature plot:

```r
featurePlot(x=training[,c("gyros_forearm_x","gyros_forearm_y","gyros_forearm_z")],
            y = training$classe,
            plot="pairs")
```

![](course_project_files/figure-html/featureplot-1.png)<!-- -->

Then, we plotted gyros forearm y versus x for each class:

```r
qplot(gyros_forearm_x,gyros_forearm_y,colour=classe,
      data=training, xlim=c(-5,5), ylim=c(-20,20))
```

```
## Warning: Removed 1 rows containing missing values (geom_point).
```

![](course_project_files/figure-html/featureplotclass-1.png)<!-- -->

As we can see more predictors are required to make the data more discriminable.

#### Building a classification model

To create our model, we first used all 53 features for the classification.

In all the trainings we standardized (z-score) the data.

We first use LDA classifier as it is a linear model and simple. It does not have hyperparameters to be tuned and it is fast to train. 10-fold cross validation was used to train the model on 9 folds and report model accuracy on a held-out test fold.

```r
set.seed(32343)
ldaFit <- train(classe~ ., data=training, method="lda", preProcess=c("center","scale"), trControl=trainControl(method="cv", number=10))
```
Next, we used Random Forests to train the model. The reason for this choice is high accuracy of this algorithm for complicated data. However, Random forests is time-consuming to train. Therefore, we used cache = TRUE in the r markdown. Here we used 10 fold cross validation too. 

```r
set.seed(1235)
rfFit <- train(classe~ .,data=training, method="rf",
                preProcess=c("center","scale"),
                trControl=trainControl(method="cv", number=10))
```
The model summary and its performance are:

```r
print(ldaFit)
```

```
## Linear Discriminant Analysis 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (57), scaled (57) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17659, 17661, 17660, 17660, 17660, 17661, ... 
## Resampling results:
## 
##   Accuracy  Kappa    
##   0.732188  0.6609137
```

```r
print(rfFit)
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (57), scaled (57) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17660, 17660, 17660, 17660, 17659, 17659, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9948527  0.9934886
##   29    0.9950055  0.9936821
##   57    0.9900109  0.9873635
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 29.
```
As we see from the results, validation accuracy of Random Forests is >99%. This means that applying the model to unseen folds predicted correct classes with >99% accuracy.

Then, in order to use fewer but more important features for the training, we made a feature extraction (dimension reduction) step with PCA to remove correlated variables. This transformed data into a new feature space. So, we repeated the training as above after this preprocessing step:

The model summary and its performance are:

```r
print(prfFit)
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (57), scaled (57), principal component
##  signal extraction (57) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17661, 17659, 17659, 17660, 17659, 17659, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9837426  0.9794334
##   29    0.9743145  0.9675058
##   57    0.9749261  0.9682770
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```
As we can see from the results, Random forests has the best performance among others. Application of PCA just slightly reduce the performance <2% while saves memory and time.

#### Prediction
We used our three models to predict the testing data:

```r
plda <- predict(ldaFit,testing) # LDA prediction
prf <- predict(rfFit,testing) # RF prediction
pprf <- predict(prfFit,testing) # RF with PCA prediction
print(plda)
```

```
##  [1] B A B A A C D D A A D A B A B A A B B B
## Levels: A B C D E
```

```r
print(prf)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
print(pprf)
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Discussion

Instead of generalization error, here we discussed the generalization performance (accuracy) which means that how good our model works instead of how bad it works. Test data are unseen data and in this problem, they are not labeled. So, we cannot calculate confusion matrix for our final predictions. However, the evaluation of model is based on the cross validation accuracy which is a kind of test accuracy.

Since we have 19622 samples in the training set, k = 10 fold will not produce large bias or large variance. In each iteration, almost 1962 samples are used as test data and 17660 samples for training which is a reasonable choice here. In practice, k = 5 or k = 10, have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.

Although, standardization of data is more important for operation of LDA, we always standardized data as a preprocessing step to scale the variables to the same range of values.

In addition, since the cross validation performance has been reported, we are mainly reporting out of sample performance. If this performance is low, it means that the model has been overfitted. But, since our performance on the validation set is high (for RF), our model has not been overfitted.

An important point to be mentioned is about the factor variable user_name. The measures can differ over subjects. In our model, we have considered the effect of this categorical variable in the prediction. However, if we remove it from the model, the model cannot capture between-subject differences and the accuracy will decrease. We tested it using LDA and performance was reduced by 3% (not shown). 
