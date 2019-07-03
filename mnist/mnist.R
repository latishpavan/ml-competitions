library(keras)
library(caret)

path <- "E:/machine learning competitions/kaggle/mnist"
setwd(path)

trainData <- read.csv("train.csv", header = TRUE, stringsAsFactors = F)
testData <- read.csv("test.csv", stringsAsFactors = FALSE)

index <- createDataPartition(trainData$label, p = 0.75, list = FALSE)
train <- trainData[index, ]
test <- trainData[-index, ]

outcome <- "label"
x_train <- train[, !colnames(train) %in% outcome]
y_train <- train[, outcome]
x_test <- test[, !colnames(test) %in% outcome]
y_test <- test[, outcome]

maxIntensity <- 255
x_train <- x_train / maxIntensity
x_test <- x_test / maxIntensity

y_train <- to_categorical(y_train, num_classes = 10)
y_test <- to_categorical(y_test, num_classes = 10)

model <- keras_model_sequential()
model %>%
    layer_dense(units = 256, activation = 'relu', 
                input_shape = c(784)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu')