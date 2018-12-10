# setwd("~/Documents/Code/secretSharer/")

library(ggplot2)
library(data.table)
library(sqldf)

data <- fread("experimentalResults.csv")

# cast as relative exposure
data$exposure = data$exposure / log(data$randomnessSpace^2, 2)

ggplot(data,
       aes(x = numEpochs,
           y = exposure,
          color = as.factor(batchSize))) + 
  geom_point(size = 4,
             position = position_jitter(width = 0.05, height = 0.01),#, height = 0.1),
             alpha = 0.6) + 
  scale_x_log10() +
  labs(title = "The number of training epochs had the only unambiguous effect on exposure",
       x = "Training epochs (log10)",
       y = "Relative exposure",
       color = "Batch size")


ggplot(data[data$numTrueSecrets < 10],
       aes(x = numTrueSecrets,
           y = exposure,
           color = as.factor(batchSize))) + 
  geom_point(size = 4,
             position = position_jitter(width = 0.2),#, height = 0.1),
             alpha = 0.6) #+ 
#geom_smooth(method = "lm")




ggplot(data[data$numFalseSecrets > 0],
       aes(x = numTrueSecrets,
           y = (numFalseSecrets / numTrueSecrets),
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 4,
             position = position_jitter(width = 0.5, height = 0.2)) + 
  #geom_smooth(method = "lm") 
  scale_color_continuous(low = "darkred",
                         high = "blue")

ggplot(data,
       aes(x = numEpochs,
           y = numTrueSecrets,
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 4,
             position = position_jitter(width = 0.05, height = 0.05)) + 
  scale_color_continuous(low = "darkred",
                         high = "blue") +
  scale_x_log10() +
  scale_y_log10() +
  labs(title = "At their extremes, training epochs and inserted secrets can make up for one another",
       x = "Training epochs (log10)",
       y = "Copies of secret inserted (log10)",
       color = "Relative exposure")


ggplot(data[data$numFalseSecrets > 0],
       aes(x = numFalseSecrets,
           y = numEpochs,
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 4,
             position = position_jitter(width = 0.5, height = 0.2)) + 
  scale_color_continuous(low = "darkred",
                         high = "blue")



ggplot(data[data$secretPrefixLength != 4],
       aes(x = secretPrefixLength,
           y = exposure,
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 4,
             position = position_jitter(width = 0.5, height = 0.2)) + 
  scale_color_continuous(low = "darkred",
                         high = "darkblue") +
  geom_smooth(method = "lm") 

sqldf("
SELECT 
  secretPrefixLength,
  COUNT(*),
  AVG(exposure) AS avg
FROM data
GROUP BY secretPrefixLength
ORDER BY secretPrefixLength
")$avg


ggplot(data[data$numTrueSecrets == 1],
       aes(x = numFalseSecrets,
           y = exposure,
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 4) +#,
             #position = position_jitter(width = 0.5, height = 0.2)) + 
  scale_color_continuous(low = "darkred",
                         high = "blue") + 
  xlim(0, 20)


sqldf("
SELECT 
      numTrueSecrets,
      COUNT(*),
      AVG(exposure) AS avg
      FROM data
      GROUP BY numTrueSecrets
      ORDER BY numTrueSecrets
      ")

sqldf("
SELECT 
      numEpochs,
      COUNT(*),
      AVG(exposure) AS avg
      FROM data
      GROUP BY numEpochs
      ORDER BY numEpochs
      ")

sqldf("
SELECT 
      batchSize,
      COUNT(*),
      AVG(exposure) AS avg
      FROM data
      GROUP BY batchSize
      ORDER BY batchSize
      ")