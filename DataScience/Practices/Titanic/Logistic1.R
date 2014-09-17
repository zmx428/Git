#read data from github
readData = function(path.name, file.name, column.types, missing.types)
{
  read.csv( url( paste(path.name, file.name, sep="") ), 
            colClasses=column.types,
            na.strings=missing.types )
}
Titanic.path = "https://raw.githubusercontent.com/zmx428/Git/master/DataScience/Practices/Titanic/"
train.data.file = "train.csv"
test.data.file = "test.csv"
missing.types = c("NA", "")
train.column.types = c('integer',   # PassengerId
                       'factor',    # Survived 
                       'factor',    # Pclass
                       'character', # Name
                       'factor',    # Sex
                       'numeric',   # Age
                       'integer',   # SibSp
                       'integer',   # Parch
                       'character', # Ticket
                       'numeric',   # Fare
                       'character', # Cabin
                       'factor'     # Embarked
)
url( paste(Titanic.path, train.data.file, sep=""))

test.column.types = train.column.types[-2]     # # no Survived column in test.csv
train.raw <- readData(Titanic.path, train.data.file, 
                      train.column.types, missing.types)
df.train <- train.raw

test.raw <- readData(Titanic.path, test.data.file, 
                     test.column.types, missing.types)
df.infer <- test.raw

attach(df.train)
#Visualize data
plot(Survived,Pclass)
#Visualize data
plot(Survived,Age)
plot(Pclass,Age)

# filling missing Age with median of Pclass
sub = df.train$Age[df.train$Pclass == 1]
med = median(sub, na.rm = TRUE)
df.train$Age[df.train$Pclass == 1][is.na(df.train$Age[df.train$Pclass == 1])] = med
sub = df.train$Age[df.train$Pclass == 2]
med = median(sub, na.rm = TRUE)
df.train$Age[df.train$Pclass == 2][is.na(df.train$Age[df.train$Pclass == 2])] = med
sub = df.train$Age[df.train$Pclass == 3]
med = median(sub, na.rm = TRUE)
df.train$Age[df.train$Pclass == 3][is.na(df.train$Age[df.train$Pclass == 3])] = med

#split data
library(caTools)
set.seed(2000)
split = sample.split(df.train$Survived, SplitRatio = 0.7)
train = subset(df.train, split == TRUE)
test = subset(df.train, split == FALSE)

# define judge function
logisticPerformance = function(train.model, testset, dependent.variable)
{
  predLog1 = predict(train.model, newdata = testset, type = "response")
  confusionLog = table(dependent.variable, predLog1 >= 0.5)
  # accurancy
  ACC = (confusionLog[1,1]+confusionLog[2,2])/sum(confusionLog)
  predROCR = prediction(predLog1, dependent.variable)
  perfROCR = performance(predROCR, "tpr", "fpr")
  plot(perfROCR, colorize=TRUE)
  # Compute AUC
  AUC = performance(predROCR, "auc")@y.values
  return (c("ACC", ACC, "AUC", AUC))
}
RandomForestPerformance = function(train.model, testset, dependent.variable)
{
  predRF = predict(modRF,  newdata = testset, type = "prob")[,2]
  confusionLog = table(dependent.variable, predRF >= 0.5)
  # accurancy
  ACC = (confusionLog[1,1]+confusionLog[2,2])/sum(confusionLog)
  predROCR = prediction(predRF, dependent.variable)
  perfROCR = performance(predROCR, "tpr", "fpr")
  plot(perfROCR, colorize=TRUE)
  # Compute AUC
  AUC = performance(predROCR, "auc")@y.values
  return (c("ACC", ACC, "AUC", AUC))
}

#train
log_train = glm(Survived ~ Sex + Pclass + Age + Fare + SibSp + Parch, data = train, family = "binomial")
library(ROCR)
logisticPerformance(log_train, test, test$Survived)

#remove variables to improve the performance of logistic regression
log_train_update = step(log_train, direction = "backward")
logisticPerformance(log_train_update, test, test$Survived)

# random forest train
library(randomForest)
#Random Forest
set.seed(144)
modRF = randomForest(Survived ~ Sex + Pclass + Age + Fare + SibSp, data = train, ntree = 1500, mtry = 2)
#This code produces a chart that for each variable measures the number of times that variable was selected for splitting (the value on the x-axis).
vu = varUsed(modRF, count=TRUE)
vusorted = sort(vu, decreasing = FALSE, index.return = TRUE)
dotchart(vusorted$x, names(modRF$forest$xlevels[vusorted$ix]))
# calculate impurity
varImpPlot(modRF)
(modRF$confusion[1,1]+modRF$confusion[2,2])/sum(modRF$confusion)
#importance
round(importance(modRF), 2)
RandomForestPerformance(modRF, test, test$Survived)


#impute df.infer
sub = df.infer$Age[df.infer$Pclass == 1]
med = median(sub, na.rm = TRUE)
df.infer$Age[df.infer$Pclass == 1][is.na(df.infer$Age[df.infer$Pclass == 1])] = med
sub = df.infer$Age[df.infer$Pclass == 2]
med = median(sub, na.rm = TRUE)
df.infer$Age[df.infer$Pclass == 2][is.na(df.infer$Age[df.infer$Pclass == 2])] = med
sub = df.infer$Age[df.infer$Pclass == 3]
med = median(sub, na.rm = TRUE)
df.infer$Age[df.infer$Pclass == 3][is.na(df.infer$Age[df.infer$Pclass == 3])] = med
df.infer$Fare[is.na(df.infer$Fare)] = median(df.infer$Fare, na.rm = TRUE)

#submission
setwd("D:/Data/Me/My Work/Programming/Data Science/Practices/Titanic")
#LR
my.predictions = predict(log_train_update, newdata = df.infer, type = "response")
#RF
my.predictions = predict(modRF, newdata = df.infer, type = "prob")[,2]
threshold = 0.5
my.predictions[my.predictions >= threshold] = 1
my.predictions[my.predictions < threshold] = 0
submission = data.frame(PassengerId = test.raw$PassengerId, Survived = my.predictions)
write.csv(submission, "submission.csv", row.names=FALSE)

#temp operation
show = my.predictions
summary(show)
str(show)