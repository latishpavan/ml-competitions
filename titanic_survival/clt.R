num <- 300
size <- 40

sampMeans <- numeric(num)

for(i in seq_len(num)){
    s <- sample(x, size)
    sampMeans[i] <- mean(s)
}

hist(sampMeans, breaks = 100)
