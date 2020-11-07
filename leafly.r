install.packages("softImpute")
install.packages("ranger")
install.packages("tidyverse")
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("tm.plugin.webmining")
install.packages("gapminder")
install.packages(c('rpart','rpart.plot'))
install.packages('e1071')
install.packages('caret')
install.packages(c("ROCR", "caTools", "MASS"))
library(GGally)
library(ROCR)
library(nnet)
library(gbm)
library(softImpute)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)
library(reshape2)
library(ggplot2)
library(tm.plugin.webmining)
library(tm)
library(SnowballC)
library(wordcloud)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(boot)
library(MASS)
library(gapminder)
library(plyr)

# data is dataframe with 25,675 observations (reviews) 
data = read.csv("UserRatingsReviews.csv")

# kag is a dataframe with all 2351 available strains of cannabis, including flavors and effects for each 
kag = read.csv("cannabis.csv")

### C L E A N I N G ###

# selecting specific columns and renaming them
cann = data[,c(3, 1, 4, 5)]
names(cann) <- c("username", "strain", "rating", "review")
names(kag) = c("strain", "Type", "Rating", "Effects", "Flavor", "Description")

# cleaning data: removing 'Anonymous', '', 'null', 'Null', NAs
cann <- cann[cann$rating != '',]
cann <- cann[!is.nan(cann$rating),]
cann = cann[!is.infinite(cann$rating),]
cann = cann[cann$rating != 'null',]
cann = cann[cann$rating != 'Null',]
cann = cann[!is.na(cann$rating),]
cann = cann[!is.infinite(cann$review),]
cann = cann[cann$username != 'Anonymous',]
cann = cann[cann$username != '',]
cann = cann[!is.na(cann$username),]
cann = cann[!is.infinite(cann$username),]

# we only want users who have done at least 3 ratings
# cann now has 4929 observations
fre= count(cann$username)
fre = fre[fre$freq > 2,]
names(fre) <- c("username", "freq")
joined = inner_join(by="username", cann, fre)
joined = joined[joined$freq > 2,]
joined = joined[order(joined$username),]
cann = joined

# assign each user and strain an ID 
cann <- transform(cann, userID = as.numeric(interaction(username, drop=TRUE)))
cann <- transform(cann, strainID = as.numeric(interaction(strain, drop=TRUE)))

# set up for collalborative filtering
cann$userID = as.integer(cann$userID)
cann$strainID = as.integer(cann$strainID)
cann$review = as.character(cann$review)

# split into strains, users, and ratings for simplicity
strains = cann[,c(2, 7)]
strains = unique(strains)
users = cann[,c(1, 6)]
users = unique(users)
rating = cann[,c(3, 6, 7)]

# cleaned kag for clustering
# merge effects and flavor into one column, replacing commas with a space
kag$efflave =  paste(kag$Effects, " ", kag$Flavor)
clus = kag[,c(1, 2, 4, 5, 7)]
clus[] <- lapply(clus, function(x) gsub("[,]", " ", x))

# we want to be able to predict the rating from the effects and flavor
# join so that each strain in cann has its corresponding effects
nlp = left_join(by="strain", cann, kag)
nlp <- lapply(nlp, function(x) gsub("[,]", " ", x))
cann = nlp

# to corpus
corpus = Corpus(VectorSource(cann$efflave))
corpusClus = Corpus(VectorSource(clus$efflave))
corpusFlav = Corpus(VectorSource(clus$Flavor))
corpusEff = Corpus(VectorSource(clus$Effects))

# made all the letters lowercase
corpus = tm_map(corpus, tolower)
corpusClus= tm_map(corpusClus, tolower)
corpusFlav = tm_map(corpusFlav, tolower)
corpusEff = tm_map(corpusEff, tolower)

# removed punctuation
corpus = tm_map(corpus, removePunctuation)
corpusClus = tm_map(corpusClus, removePunctuation)
corpusFllav = tm_map(corpusFlav, removePunctuation)
corpusEff = tm_map(corpusEff, removePunctuation)

# looked at the corpus to see which words were likely included in a high proportion of the Bodies. The most obvious was "strain"
# can skip this for corpusClus
strwrap(corpus[[1]])
corpus = tm_map(corpus, removeWords, c("this", "strain", stopwords("english")))
strwrap(corpus[[1]])

# stemed the document 
# can skip this for corpusClus
corpus = tm_map(corpus, stemDocument)
strwrap(corpus[[1]])

# checked the number of terms
# converts corpusClus into a DocumentTermMatrix
frequencies = DocumentTermMatrix(corpus)
frequencies
frequenciesClus = DocumentTermMatrix(corpusClus) # 37 terms
frequenciesFlav = DocumentTermMatrix(corpusFlav)
frequenciesEff = DocumentTermMatrix(corpusEff)

# accounted for sparsity, trying to limit the DocumentTermMatrix to about 200 terms (229)
# can skip this for corpusClus
sparse = removeSparseTerms(frequencies, 0.985)
sparse

#created data frame from DocumentTermMatrix
cannTM = as.data.frame(as.matrix(sparse))
colnames(cannTM) = make.names(colnames(cannTM))
cannTM$rating = cann$rating

clusTM = as.data.frame(as.matrix(frequenciesClus))
colnames(clusTM) = make.names(colnames(clusTM))
clusTM$type = as.numeric(clus$Type == "sativa")

flavTM = as.data.frame(as.matrix(frequenciesFlav))
colnames(flavTM) = make.names(colnames(flavTM))
flavTM$type = as.numeric(clus$Type == "sativa")

effTM = as.data.frame(as.matrix(frequenciesEff))
colnames(effTM) = make.names(colnames(effTM))
effTM$type = as.numeric(clus$Type == "sativa")

#clusTM$Strain = clus$Strain

#splitting the data into training, valA, valB, and test for collaborative filtering model
set.seed(123)
train.ids.cann <- sample(nrow(cannTM), 0.84*nrow(cannTM))
train.rating <- rating[train.ids.cann,]
test.rating <- rating[-train.ids.cann,]
val.idsA.rating <- sample(nrow(test.rating), (8/16)*nrow(test.rating))
valA.rating <- test.rating[val.idsA.rating,]
test.rating <- test.rating[-val.idsA.rating,]
val.idsB.rating <- sample(nrow(valA.rating), (4/8)*nrow(valA.rating))
valB.rating <- valA.rating[val.idsB.rating,]
valA.rating <- valA.rating[-val.idsB.rating,]

train.rating$rating = as.numeric(train.rating$rating) - 1
test.rating$rating = as.numeric(test.rating$rating) - 1
valA.rating$rating = as.numeric(valA.rating$rating) - 1
valB.rating$rating = as.numeric(valB.rating$rating) - 1

# splitting the data into training, test, and valB matrices. 
# these must be different from the dataframes in  the previous block 
# because they will be used to predict rating from the words used in each review
# idea is to blend softImpute model with other models (boosting, random forest, etc.)
cannTrain = cannTM[train.ids.cann,]
cannTest = cannTM[-train.ids.cann,]
cannTest = cannTest[-val.idsA.rating,]
cannValB = cannTM[val.idsB.rating,]

cannTrain$rating = as.numeric(cannTrain$rating)   # turned to 6? 
cannTest$rating = as.numeric(cannTest$rating) 
cannTrain$rating

# created new identical matrices that ratings into factors 
# model prediction is categorical (except for linear regression)
cannTrain.factor = cannTrain
cannTrain.factor$rating = as.factor(cannTrain.factor$rating)
cannTrain.factor$rating

cannTest.factor = cannTest
cannTest.factor$rating = as.factor(cannTest.factor$rating)

cannValB.factor = cannValB
cannValB.factor$rating = as.factor(cannValB.factor$rating)

##### B A S E L I N E #####

baseAccuracyTrain = sum(round(train.rating$rating) == 4)/4140 # .3495 (mean rating for train is 4.341)
baseAccuracyTrain
baseAccuracyTest = sum(round(test.rating$rating) == 4)/395 # .3544 
baseAccuracyTest

### B L E N D I N G    M O D E L S ###

# RANDOM FOREST 
mod.rf = ranger(rating ~ ., 
                data = cannTrain.factor, 
                mtry = floor((ncol(cannTrain)/2)), 
                num.trees = 100,
                verbose = TRUE)

predict.rf = predict(mod.rf, data = cannTest.factor)$predictions
table(cannTest$rating, predict.rf)
tableAccuracy(cannTest$rating, predict.rf) #.4987

rf_df = data.frame(labels = cannTest$rating, predictions = predict.rf)
RF_boot = boot(rf_df, boot_all_metrics, R = 1000)
RF_boot
boot.ci(RF_boot, index = 1, type = "basic") # 95% cofidence interval for accuracy is (.4532, .9797)

# CART (CV) 
train.cart = train(rating ~ .,
                   data = cannTrain.factor,
                   method = "rpart",
                   tuneGrid = data.frame(cp=seq(0, 0.4, 0.002)),
                   trControl = trainControl(method="cv", number=10))
train.cart
train.cart$results

ggplot(train.cart$results, aes(x = cp, y = Accuracy)) + geom_point(size = 2) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mod.cart = train.cart$finalModel
prp(mod.cart)

predict.cart = predict(mod.cart, newdata = cannTest.factor, type="class")
table(cannTest.factor$rating, predict.cart)
tableAccuracy(cannTest$rating, predict.cart)  #.4962

cart_df = data.frame(labels = cannTest$rating, predictions = predict.cart)
CART_boot = boot(cart_df, boot_all_metrics, R = 1000)
CART_boot
boot.ci(CART_boot, index = 1, type = "basic") # 95% cofidence interval for accuracy is (.4506, .9747)

# LOGISTIC REGRESSION 
mod.log <- nnet::multinom(rating ~ ., data = cannTrain.factor)
summary(mod.log)

predict.log = predict(mod.log, newdata = cannTest.factor, type = "class") 
table(cannTest.factor$rating, predict.log)
tableAccuracy(cannTest$rating, predict.log) # .4835

log_df = data.frame(labels = cannTest$rating, predictions =  predict.log)
Log_boot = boot(log_df, boot_all_metrics, R = 1000)
Log_boot
boot.ci(Log_boot, index = 1, type = "basic") # 95% cofidence interval for accuracy is (.4329, .9443)

# BOOSTING (CV) 
tGrid = expand.grid(n.trees = (1:100), interaction.depth = c(1,3,6, 9, 12),
                    shrinkage = 0.01, n.minobsinnode = 10)
train.boost <- train(rating ~ .,
                     data = cannTrain.factor,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "Accuracy",
                     distribution = "multinomial")

train.boost
train.boost$results

ggplot(train.boost$results, aes(x = n.trees, y = Accuracy, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV Accuracy") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

mod.boost = train.boost$finalModel

cannTest.mm = as.data.frame(model.matrix(rating ~ ., data = cannTest.factor))
predict.boost = predict(mod.boost, newdata = cannTest.mm, n.trees = 100, type = "response")
pred = apply(predict.boost, 1, which.max)
predict.boost = factor(pred, levels = c(1,2,3,4,5), labels = c("1", "2", "3", "4", "5"))

table(cannTest$rating, predict.boost)
tableAccuracy(cannTest$rating, predict.boost) #.4785 

boost_df = data.frame(labels = cannTest$rating, predictions = predict.boost)
Boost_boot = boot(boost_df, boot_all_metrics, R = 1000)
Boost_boot
boot.ci(Boost_boot, index = 1, type = "basic") # 95% confidence interval for accuracy is (0.4329,  0.9139) 

# LINEAR DISCRIMINANT ANALYSIS 
mod.lda = lda(rating ~ ., data = cannTrain.factor)

predict.lda = predict(mod.lda, newdata = cannTest)$class
table(cannTest$rating, predict.lda)
tableAccuracy(cannTest$rating, predict.lda) #.4861

lda_df = data.frame(labels = cannTest$rating, predictions = predict.lda)
Lda_boot = boot(lda_df, boot_all_metrics, R = 1000)
Lda_boot
boot.ci(Lda_boot, index = 1, type = "basic") # 95% cofidence interval for accuracy is (.4354, .9367)

# LINEAR REGRESSION 
mod.lin <- lm(rating ~ ., data = cannTrain)
summary(mod.lin)

# round to turn back into integer
predict.lin <- predict(mod.lin, newdata = cannTest) #%>% pmin(5) %>% pmax(1))
predict.lin = round(predict.lin)

table(cannTest$rating, predict.lin)
tableAccuracy(cannTest$rating, predict.lin) # .35 (hand calculated)

lin_df = data.frame(labels = cannTest$rating, predictions = predict.lin)
Lin_boot = boot(lda_df, boot_all_metrics, R = 1000)
Lin_boot
boot.ci(Lin_boot, index = 1, type = "basic") # 95% cofidence interval for accuracy is (.4030, .6046)

### C O L L A B O R A T I V E    F I L T E R I N G ###

# create Incomplete matrix
mat.train <- Incomplete(train.rating$userID, train.rating$strainID, train.rating$rating)

# cross validating for rank
acc.vals.rating = rep(NA, 30)
for (rnk in seq_len(30)) {
  mod <- softImpute(mat.train, rank.max = rnk, lambda = 0, maxit = 10000000)
  preds <- impute(mod, valA.rating$userID, valA.rating$strainID) %>% pmin(5) %>% pmax(1) 
  acc.vals.rating[rnk] <- tableAccuracy(valA.rating$rating, round(preds)) 
}

acc.val.df <- data.frame(rnk = seq_len(30), acc = acc.vals.rating)
ggplot(acc.val.df, aes(x = rnk, y = acc)) + geom_point(size = 3) + 
  ylab("Validation Accuracy") + xlab("Number of Archetypal Users") + 
  theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

# using CV, the k value which increases accuracy is 1 
# only 1 kind of smoker? 

mod.final <- softImpute(mat.train, rank.max = 1, lambda = 0, maxit = 10000000)
preds.final <- impute(mod.final, test.rating$userID, test.rating$strainID) %>% pmin(5) %>% pmax(1)

table(cannTest$rating, round(preds.final)) 
tableAccuracy(cannTest.factor$rating, round(preds.final)) # .2810 (ouch)

### B L E N D I N G ###

cannValB.mm = as.data.frame(model.matrix(rating ~ ., data = cannValB.factor))

val.preds.cf <- impute(mod.final, valB.rating$userID, valB.rating$strainID) %>% pmin(5) %>% pmax(1)
val.preds.boost <- predict(mod.boost, newdata = cannValB.mm, n.trees = 100, type = "response")
#val.preds.lda <- predict(mod.lda, newdata = cannValB.factor)$class
val.preds.cart = predict(mod.cart, newdata = cannValB.factor, type="class")
val.preds.rf <- predict(mod.rf, data = cannValB.factor)$predictions
val.preds.log = predict(mod.log, newdata = cannValB.factor, type = "class")
val.preds.lin = predict(mod.lin, newdata = cannValB)

pred1 = apply(val.preds.boost, 1, which.max)
val.preds.boost = factor(pred1, levels = c(1,2,3,4,5), labels = c("1", "2", "3", "4", "5"))

# build validation set data frame
val.blending_df = data.frame(rating = valB.rating$rating, cf_preds = val.preds.cf, rf_preds = val.preds.rf, 
                             log_preds = val.preds.log, cart_preds = val.preds.cart, # lda_preds = val.preds.lda, 
                             boost_preds = val.preds.boost, lin_preds = val.preds.lin)

# train blended model
blend.mod = lm(rating ~ ., data = val.blending_df) # -1: no intercept
summary(blend.mod)

# Get predictions on test set
test.preds.cf <- impute(mod.final, test.rating$userID, test.rating$strainID) %>% pmin(5) %>% pmax(1)
test.preds.boost <- predict(mod.boost, newdata = cannTest.mm, n.trees = 100, type = "response")
#test.preds.lda = predict(mod.lda, newdata = cannTest.factor)$class
test.preds.cart = predict(mod.cart, newdata = cannTest.factor, type = "class")
test.preds.rf <- predict(mod.rf, data = cannTest.factor)$predictions
test.preds.log = predict(mod.log, newdata = cannTest.factor, type = "class")
test.preds.lin = predict(mod.lin, newdata = cannTest)

pred2 = apply(test.preds.boost, 1, which.max)
test.preds.boost = factor(pred2, levels = c(1,2,3,4,5), labels = c("1", "2", "3", "4", "5"))

# build test set data frame
test.blending_df = data.frame(rating = test.rating$rating, cf_preds = test.preds.cf, rf_preds = test.preds.rf, 
                              log_preds = test.preds.log, cart_preds = test.preds.cart, # lda_preds = test.preds.lda,
                              boost_preds = test.preds.boost, lin_preds = test.preds.lin)
nrow(test.blending_df)

# predict based on blended model
test.preds.blend <- predict(blend.mod, newdata = test.blending_df)  %>% pmin(5) %>% pmax(1)
length(test.preds.blend)
rounded = ceiling(test.preds.blend) # is ceiling ok?
length(rounded)

table(test.rating$rating, rounded)
tableAccuracy(test.rating$rating, test.preds.blend) # .4911 (hand calculated)

# S U M M A R Y #
# accuracy of blended model was higher than baseline
# this is largely due to the non collaborative filtering models
# collaborative filtering itself performed worse than the baseling
# likely due to the sparsity of the Incomplete matrix 
# even after removing users with fewer than 3 reviews, most users still had not reviewed many strains
# important to know that cf model ratings were rounded (since other models predicted 1-5 categorically)

# another idea is to cluster strains by their effects and flavors

### C L U S T E R I N G ###

cluster = clusTM
pp <- preProcess(cluster, method=c("center", "scale"))

cluster.scaled <- predict(pp, cluster) 

# K-Means #

# selecting k 
dat <- data.frame(k = 1:200)
dat$SS <- sapply(dat$k, function(k) {
  set.seed(123)
  kmeans(cluster.scaled, iter.max=100, k)$tot.withinss
})

# plot
ggplot(dat, aes(x=k, y=SS)) +
  geom_line() +
  xlab("Number of Clusters (k)") +
  ylab("Within-Cluster Sum of Squares") +
  geom_vline(xintercept =0, color = "blue")

# best kMeans clustering
km <- kmeans(cluster.scaled, iter.max=100, 55) # eyeballed k value

# hierarchical clustering #

# compute all-pair euclidian distances between our observations
d <- dist(cluster.scaled)

# creates hierarchical clustering
mod.hclust <- hclust(d, method="ward.D2")

plot(mod.hclust, labels = FALSE)

# removing extra text 
plot(mod.hclust, labels=F, xlab=NA, ylab="Dissimilarity",sub=NA, main=NA)

dat.hc.cluster <- data.frame(k = seq_along(mod.hclust$height),
                             dissimilarity = rev(mod.hclust$height))

# scree 
ggplot(dat.hc.cluster, aes(x=k, y=dissimilarity)) +
  geom_line()+
  xlab("Number of Clusters") +
  ylab("Dissimilarity")

# zoom in
ggplot(dat.hc.cluster, aes(x=k, y=dissimilarity)) +
  geom_line()+
  xlab("Number of Clusters") +
  ylab("Dissimilarity") + 
  xlim(0,100)

# now that we have k, we can construct the clusters
assignments <- cutree(mod.hclust, k = 49)
rev(mod.hclust$height)[48]

# store it into our data frame
cluster.scaled$cluster = assignments
clus$cluster_both = assignments

# get centroids
cluster.scaled %>% group_by(clust2) %>%
  summarize_all(funs(mean))

# size of each cluster
table(assignments)

# by_cluster is a dataframe ordered by cluster
# cluster 6 has efflave that contains "None" (missing effects and flavor)
by_cluster_both = clus[order(clus$cluster),]

# clustering using only flavor

flavor = flavTM
pp <- preProcess(flavor, method=c("center", "scale"))

flavor.scaled <- predict(pp, flavor) 

d2 <- dist(flavor.scaled)

# creates hierarchical clustering
mod.hclust2 <- hclust(d2, method="ward.D2")

plot(mod.hclust2, labels = FALSE)

# removing extra text 
plot(mod.hclust2, labels=F, xlab=NA, ylab="Dissimilarity",sub=NA, main=NA)

dat.hc.cluster2 <- data.frame(k = seq_along(mod.hclust2$height),
                             dissimilarity = rev(mod.hclust2$height))

# scree 
ggplot(dat.hc.cluster2, aes(x=k, y=dissimilarity)) +
  geom_line()+
  xlab("Number of Clusters") +
  ylab("Dissimilarity")

# zoom in
ggplot(dat.hc.cluster2, aes(x=k, y=dissimilarity)) +
  geom_line()+
  xlab("Number of Clusters") +
  ylab("Dissimilarity") + 
  xlim(0,100)

# now that we have k, we can construct the clusters
assignments2 <- cutree(mod.hclust2, k = 49)
rev(mod.hclust2$height)[48]

# store it into our data frame
flavor.scaled$cluster = assignments2
flavor$cluster = assignments2
clus$cluster_flavor = assignments2

# get centroids
flavor.scaled %>% group_by(cluster) %>%
  summarize_all(funs(mean))

# size of each cluster
table(assignments2)

# by_cluster is a dataframe ordered by cluster
# cluster 6 has efflave that contains "None" (missing effects and flavor)
by_cluster_flavor = clus[order(clus$cluster_flavor),]

# clustering using only effects

effects = effTM
pp <- preProcess(effects, method=c("center", "scale"))

effects.scaled <- predict(pp, effects) 

d3 <- dist(effects.scaled)

# creates hierarchical clustering
mod.hclust3 <- hclust(d3, method="ward.D2")

plot(mod.hclust3, labels = FALSE)

# removing extra text 
plot(mod.hclust3, labels=F, xlab=NA, ylab="Dissimilarity",sub=NA, main=NA)

dat.hc.cluster3 <- data.frame(k = seq_along(mod.hclust3$height),
                              dissimilarity = rev(mod.hclust3$height))

# scree 
ggplot(dat.hc.cluster3, aes(x=k, y=dissimilarity)) +
  geom_line()+
  xlab("Number of Clusters") +
  ylab("Dissimilarity")

# zoom in
ggplot(dat.hc.cluster3, aes(x=k, y=dissimilarity)) +
  geom_line()+
  xlab("Number of Clusters") +
  ylab("Dissimilarity") + 
  xlim(0,100)

# now that we have k, we can construct the clusters
assignments3 <- cutree(mod.hclust3, k = 20)
rev(mod.hclust3$height)[48]

# store it into our data frame
effects.scaled$cluster = assignments3
effects$cluster = assignments3
clus$cluster_effects = assignments3

# get centroids
effects.scaled %>% group_by(cluster) %>%
  summarize_all(funs(mean))

# size of each cluster
table(assignments3)

# by_cluster is a dataframe ordered by cluster
# cluster 6 has efflave that contains "None" (missing effects and flavor)
by_cluster_effects = clus[order(clus$cluster_effects),]

# S U M M A R Y #
# clustering did not work so well using words from scraped data reviews
# scraped data was too sparse
# taking the effects and flavors from a kaggle dataset and merging them 
# into 66 concise variables worked better
# would have been better if variables like "sweet" or "euphoric" were quantitative metrics of the variable

# FUNCTIONS
mean_squared_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  MSE <- mean((responses - predictions)^2)
  return(MSE)
}

mean_absolute_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  MAE <- mean(abs(responses - predictions))
  return(MAE)
}

OS_R_squared <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  baseline <- data$baseline[index]
  SSE <- sum((responses - predictions)^2)
  SST <- sum((responses - baseline)^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

all_metrics <- function(data, index) {
  mse <- mean_squared_error(data, index)
  mae <- mean_absolute_error(data, index)
  OSR2 <- OS_R_squared(data, index)
  return(c(mse, mae, OSR2))
}

tableAccuracy <- function(label, pred) {
  t = table(label, pred)
  a = sum(diag(t))/length(label)
  return(a)
}

tableTPR <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[2,1] + t[2,2]))
}

tableFPR <- function(label, pred) {
  t = table(label, pred)
  return(t[1,2]/(t[1,1] + t[1,2]))
}

boot_accuracy <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableAccuracy(labels, predictions))
}

boot_tpr <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableTPR(labels, predictions))
}

boot_fpr <- function(data, index) {
  labels <- data$label[index]
  predictions <- data$prediction[index]
  return(tableFPR(labels, predictions))
}

boot_all_metrics <- function(data, index) {
  acc = boot_accuracy(data, index)
  tpr = boot_tpr(data, index)
  fpr = boot_fpr(data, index)
  return(c(acc, tpr, fpr))
}

tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

