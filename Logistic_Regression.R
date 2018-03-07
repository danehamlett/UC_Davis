########################################
#   GLM Models - Logistic Regression   #
########################################

# Import primary libraries and data set
library(pscl)
library(lmtest)
library(MKmisc)
library(survey)
library(pROC)
library(ROCR)
library(ResourceSelection)
library(caret)
data(GermanCredit)

# Show the first 10 columns
str(GermanCredit[, 1:10])

# Preview the full data set
View(GermanCredit)

# Create training and test data sets - our training data set will contain 600 results
Train <- createDataPartition(GermanCredit$Class, p=0.6, list=FALSE)
training <- GermanCredit[Train,]
testing <- GermanCredit[-Train,]

# Let's select a few variables to add to our model, and go ahead and build the model
mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
                   CreditHistory.Critical,  data=training, method="glm", family="binomial")

# Let's look at the summary of our model results
summary(mod_fit)

# Estimates from logistic regression model reflect the relationship between the predictor and the
# response variable in a log-odds scale; use an exponential conversion to calculate the odds ratio
# for each predictor

# When reviewing the results, we can see that for every one unit increase in Age, the odds of having
# good credit increases by a factor of 1.01
exp(coef(mod_fit$finalModel))

# Now we can run two predictive models on the data 
predict(mod_fit, newdata=testing)
predict(mod_fit, newdata=testing, type="prob")

# Let's assess how good our model is, by running two models
mod_fit_one <- glm(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
                     CreditHistory.Critical, data=training, family="binomial")

mod_fit_two <- glm(Class ~ Age + ForeignWorker, data=training, family="binomial")

# A logistic regression model will be more effective if it demonstrates lift over an alternate model
# with fewer predictors. A likelihood ratio test will confirm this, which compares the the likelihood
# of data under the full model, against the likelihood of data under the model with fewer predictors.
# A model with fewer predictors will have a lower log likelihood (will fit less well), but it is required
# to test whether the observed differences are statistically significant.

# Null hypothesis = the second model provides a better fit
# Alternative hypothesis = the second model does not provide a better fit
# Alpha = .05

# Either method can be used below to determine that we can reject the null hypothesis in favor
# of the first model
anova(mod_fit_one, mod_fit_two, test ="Chisq")
lrtest(mod_fit_one, mod_fit_two)

# McFadden's R2 test is a proxy for R2 in linear regression
# Values range between 0 and 1, with values closer to 0 reflecting little predictive power
pR2(mod_fit_one)

# The Hosmer-Lemeshow Test gives a goodness of fit measure, after observations have been segmented
# into groups, based off of having similar predicted probabilities. Ultimately determining whether
# observed probabilities are similar to predicted probabilities of an occurrence, using a pearson
# Chi-Square test.
# Small values with large p-values indicate a good fit to the data, while large values with a p-value below
# .05 indicate a poor fit
HLgof.test(fit = fitted(mod_fit_one), obs = training$Class)

# Alternate method to test goodness of fit, using the same metric
hoslem.test(training$Class, fitted(mod_fit_one), g=10)

# The Wald test allows us to evaluate the statistical significance of each coefficient in the model
# Null hypothesis = coefficient is equal to 0
# Alternative hypothesis = coefficient is not equal to 0
# If p < .05, we are unable to reject the null hypothesis, we can be confident that removing the predictor
# will not affect the model

# We can be confident in eliminating this predictor from the model
regTermTest(mod_fit_one, "ForeignWorker")

# We cannot be confident in eliminating this predictor from the model
regTermTest(mod_fit_one, "CreditHistory.Critical")

# Let's assess the GLM variable importance for each variable
varImp(mod_fit)

# Let's test how the model performs when predicting the target variable on a sample of observations
# We can see the percentage of predicted observations was around 68%, as seen below
pred <- predict(mod_fit, newdata=testing)
accuracy <- table(pred, testing[,"Class"])
sum(diag(accuracy))/sum(accuracy)

# We can use the script below to generate a confusion matrix
pred <- predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$Class)

# ROC stands for the Receiving Operating Characteristic, which is a measure of classifier performance
# This measure assesses the proportion of positive data points that are correctly considered as positive
# and the proportion of negative data points that are mistakenly considered to be positive, and a plot
# is created to show the tradeoff between correctly and incorrectly predicting an outcome.

# We ultimately want to predict the area under the ROC curve, or the AUROC, which is a metric from .5 - 1
# with values above .80 doing a good job at discriminating the two categories that make up the
# target variable

# Compute AUC for predicting Class with the variable CreditHistory.Critical
f1 <- roc(Class ~ CreditHistory.Critical, data=training) 
plot(f1, col="red")

# Compute AUC for predicting Class with the model
prob <- predict(mod_fit_one, newdata=testing, type="response")
pred <- prediction(prob, testing$Class)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

# K-fold cross validation; used to assess how well a model predicts the target variable on 
# different subsets of the data. Data is partitioned into equally-sized subsets, while
# one fold is held out for validation.

ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

mod_fit <- train(Class ~ Age + ForeignWorker + Property.RealEstate + Housing.Own + 
                   CreditHistory.Critical,  data=GermanCredit, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)

pred <- predict(mod_fit, newdata=testing)
confusionMatrix(data=pred, testing$Class)