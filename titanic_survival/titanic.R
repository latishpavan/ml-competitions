library(caret)
library(RANN)


missing.values <- function(df) {
    for(col in names(df)) {
        print(paste(col, sum(is.na(df[[col]])), sep = " "))
    }
}


fileDir <- "E:/machine learning competitions/kaggle/titanic_survival/"
fullDataPath <- paste(fileDir, "fullData.csv", sep = "")
testPath <- paste(fileDir, "test.csv", sep = "")

setwd(fileDir)

#columns <- c("integer", "integer", "integer", "character",
#             "character", "numeric", "integer", "integer",
#             "character", "numeric", "character", "character")

fullData <- read.csv(fullDataPath, stringsAsFactors = FALSE, na.strings = c("", " "))
test <- read.csv(testPath, stringsAsFactors = FALSE, na.strings = c("", " "))

fullData$isfullData <- TRUE
test$isfullData <- FALSE
test$Survived <- 0

fullData <- rbind(fullData, test)

medianAge <- median(fullData$Age, na.rm = TRUE)
fullData$Age[is.na(fullData$Age)] <- medianAge
fullData$Sex <- as.factor(fullData$Sex)
fullData$Embarked <- as.factor(fullData$Embarked)
fullData$Survived <- as.factor(fullData$Survived)
fullData$Pclass <- as.factor(fullData$Pclass)
freq <- table(fullData$Embarked)
freqClass <- names(freq)[which.max(freq)]
fullData$Embarked[is.na(fullData$Embarked)] <- freqClass

dont.keeps <- c("Cabin", "Ticket", "Name")
trimmed.data <- fullData[, !colnames(fullData) %in% dont.keeps]
medianFare <- median(trimmed.data$Fare, na.rm = TRUE)
trimmed.data$Fare[is.na(trimmed.data$Fare)] <- medianFare

trimmed.data$Survived <- ifelse(trimmed.data$Survived == "0", 0, 1)
dmy <- dummyVars("~.", data = trimmed.data)
trimmed.data <- data.frame(predict(dmy, newdata = trimmed.data))
trimmed.data$Survived <- as.factor(trimmed.data$Survived)
fares <- trimmed.data$Fare
trimmed.data$Fare <- (fares - mean(fares)) / sqrt(var(fares))

test.data <- trimmed.data[trimmed.data$isTrainFALSE == 1, ]
train.data <- trimmed.data[trimmed.data$isTrainTRUE == 1, ]

removes <- c("PassengerId", "isTrainFALSE", "isTrainTRUE",
             "isfullDataTRUE", "isfullDataFALSE")

train.data <- train.data[, !colnames(train.data) %in% removes]

index <- createDataPartition(train.data$Survived, p = 0.75, list = F)
train.set <- train.data[index, ]
validation.set <- train.data[-index, ]

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      number = 3,
                      verbose = FALSE)

outcame.name <- "Survived"
predictors <- colnames(train.data)[!colnames(train.data) %in% outcame.name]

features <- rfe(train.set[, predictors], train.set[, outcame.name],
                rfeControl = control)

best.features <- features$optVariables
fit.control <- trainControl(method = "repeatedcv",
                            repeats = 3, 
                            number = 5)

grid <- expand.grid(mtry = c(10, 20, 30, 40))

model_rf <- train(train.set[, best.features],
                  train.set[, outcame.name],
                  method = "rf",
                  trControl = fit.control,
                  tuneLength = 10)

model_gb <- train(train.set[, best.features],
                  train.set[, outcame.name],
                  method = "gbm",
                  trControl = fit.control,
                  tuneLength = 10)

model_nn <- train(train.set[, best.features],
                  train.set[, outcame.name],
                  method = "nnet",
                  trControl = fit.control,
                  tuneLength = 10)

model_xgb <- train(train.set[, best.features],
                   train.set[, outcame.name],
                   method = "xgbTree",
                   trControl = fit.control,
                   tuneLength = 10)


model_ada <- train(train.set[, best.features],
                  train.set[, outcame.name],
                  method = "adaboost",
                  trControl = fit.control,
                  tuneLength = 10)


mode <- function(data, num = FALSE){
    freq <- which.max(table(data))
    ifelse(num, as.numeric(names(freq)), names(freq))
}

# forming ensemble of rf, gb and nn
ensemble.predict <- function(data, models){
    len <- nrow(data)
    temp <- numeric(length = len)
    
    for(model in models) {
        mod_preds <- predict.train(object = model, 
                                   data,
                                   type = "raw")
        
        temp <- rbind(temp, mod_preds)
    }
    
    temp <- temp[2:nrow(temp), ]
    predictions <- apply(temp, MARGIN = 2, mode, num = TRUE)
    return(predictions)
} 

models <- list(model_rf)
ensemble.preds <- ensemble.predict(validation.set[, best.features],
                                   models)

predictions_gb <- predict.train(object = model_gb,
                                validation.set[, best.features],
                                type = "raw")
# 
predictions_rf <- predict.train(object = model_rf,
                                validation.set[, best.features],
                                type = "raw")
# 
# predictions_nn <- predict.train(object = model_nn,
#                                 validation.set[, best.features],
#                                 type = "raw")
# 
predictions_ada <- predict.train(object = model_ada,
                                validation.set[, best.features],
                                type = "raw")
# 
confusionMatrix(validation.set[, outcame.name], predictions_gb)
confusionMatrix(validation.set[, outcame.name], predictions_rf)
# confusionMatrix(validation.set[, outcame.name], predictions_nn)
confusionMatrix(validation.set[, outcame.name], predictions_ada)

ensemble.preds <- as.factor(ensemble.preds - 1)
confusionMatrix(validation.set[, outcame.name], ensemble.preds)

final.ans <- predict.train(object = model_rf,
                           test.data[, best.features],
                           type = "raw")

submit <- data.frame(cbind(test.data$PassengerId, final.ans))
submit$final.ans <- submit$final.ans - 1
colnames(submit) <- c("PassengerId", "Survived")
write.csv(x = submit, file = "submit.csv", row.names = FALSE)
