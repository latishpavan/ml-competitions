library("caret")

dependency.test <- function(df, target, significanceLevel = 0.05) {
    result <- vector("character", ncol(df)-1)
    predictors <- names(df)[!names(df) %in% target]
    names(result) <- predictors
    
    try(
        for(col in predictors) {
            test <- chisq.test(df[[target]], df[[col]])
            if(test$p.value <= significanceLevel) {
                result[col] <- "Dependent"
            } else {
                result[col] <- "Independent"
            }
        }
    )
    return(result)
}


path <- "E:/machine learning competitions/kaggle/mushroom_classification/mushrooms.csv"

raw_data <- read.csv(path, stringsAsFactors = TRUE)
raw_data$veil.type <- as.numeric(raw_data$veil.type)
raw_data$class <- ifelse(raw_data$class == "p", 0, 1)

dmy <- dummyVars("~.", data = raw_data)
data <- data.frame(predict(dmy, newdata = raw_data))
data$class <- as.factor(data$class)

index <- createDataPartition(data$class, p = 0.75, list = FALSE)
trainSet <- data[index, ]
testSet <- data[-index, ]

control <- rfeControl(functions = rfFuncs,
                      method = "cv",
                      number =  5,
                      verbose = FALSE)

outcomeName <- "class"
predictors <- names(trainSet)[!names(trainSet) %in% outcomeName]
features <- rfe(trainSet[, predictors], trainSet[, outcomeName],
                rfeControl = control)

selectedPredictors <- c("odor.n", "spore.print.color.r", "gill.size.n", 
                        "odor.f", "gill.size.b")

fitControl <- trainControl(method = "cv",
                           number = 5
                           )

grid <- expand.grid(mtry=c(10, 15, 20))

model_rf <-  train(trainSet[, selectedPredictors], 
                   trainSet[, outcomeName], 
                   method = "rf",
                   trControl = fitControl,
                   tuneGrid = grid)

predictions <- predict.train(object = model_rf,
                             testSet[, selectedPredictors],
                             type = "raw")

confusionMatrix(predictions, testSet[, outcomeName])

