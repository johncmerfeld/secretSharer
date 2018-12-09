# setwd("~/Documents/Code/secretSharer/")

library(ggplot2)
library(data.table)

data <- fread("experimentalResults.csv")

# cast as relative exposure
data$exposure = data$exposure / log(data$randomnessSpace^2, 2)

ggplot(data[data$numEpochs < 50],
       aes(x = numEpochs,
           y = exposure)) + 
  geom_point(aes(size = as.factor(batchSize)),
             position = position_jitter(width = 0.9, height = 0.2),
             alpha = 0.6) #+ 
  #geom_smooth(method = "lm")

ggplot(data[data$numFalseSecrets > 1],
       aes(x = numTrueSecrets,
           y = (numFalseSecrets / numTrueSecrets),
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 2) + 
  #geom_smooth(method = "lm") + 
  geom_jitter(width = 0.5, height = 0.2)

ggplot(data,
       aes(x = numTrueSecrets,
           y = numEpochs,
           color = exposure)) + 
  geom_point(alpha = 0.6,
             size = 4,
             position = position_jitter(width = 0.5, height = 0.2)) + 
  scale_color_continuous(low = "darkred",
                         high = "darkblue")

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

