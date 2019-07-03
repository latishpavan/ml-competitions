df <- read.csv("https://goo.gl/j6lRXD")
table(df$treatment, df$improvement)

chisq.test(df$treatment, df$improvement, correct = FALSE)
