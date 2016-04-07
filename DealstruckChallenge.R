## Tuesday 2:30 stop around 6, had breaks.
## Thurs 10:15
# Can always use more years of data. Would be interesting if we could look at the history of 
# the same loans to see. Especially need more delinquent loans. 

## Downloaded more data, was missing FICO. Seems important. Using these got accuracy
## of around 77% ensembling LASSO, Ridge, RF models at downsamples of equal, 1.5
## and double. Ended up basically same result as downsample 1.5, not sure if ensemble
## is that useful as models get very similar results. 
library('dplyr')
library('randomForest')
library('ggplot2')
library('glmnet')
library('caret')
setwd('~/Dealstruck/')
# ----------------------------------------------------------------
#### DATA IMPORT, CLEANSING, TRANSFORMATION, EXPLORATORY ANALYSIS
# ----------------------------------------------------------------
loanData <- read.csv('LC_biz_all.csv')
# loanData <- read.csv('LoanStats3ck.csv', skip=1)

# colClasses=c("NULL", NA) null skips, na finds value
#Preliminary look at the dataset
str(loanData)
summary(loanData)

# emp_title, zip_code, and addr_state have far too many categories relative to the size of
# our dataset, as there are onyl ~500 delinquent responses. A factor with 50
# or more categories cannot be reasonably fit to that much data. purpose has only 1 leve,
# so it has no information
drops <- c('emp_title', 'purpose', 'zip_code', 'addr_state')
loanData <- loanData[, !(names(loanData) %in% drops)]

# revol_util, int_rate are percents as strings. Change to doubles.
loanData$revol_util <- as.double(gsub('%', '', loanData$revol_util))
loanData$int_rate <- as.double(gsub('%', '', loanData$int_rate))

# Checking how many NA values we have.
naCounts <- sapply(loanData, function(x) sum(is.na(x)))
naCounts[naCounts >0]

# many of the mnths_since categories are almost entirely NA, likely due to having
# no delinquency or record. We will remove these columns rather than trying to fill 
# the NA values because I don't see a valid strategy for filling them. We could use
# replace them with something like the maximum value, but I don't think that is a
# good approach. The quantities these are trying to measure are likely to be
# measured in categories like inq_last_6mths
naDrops <- c('mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
           'mths_since_recent_bc_dlq', 'mths_since_recent_inq')
loanData <- loanData[, !(names(loanData) %in% naDrops)]

# For the remaining NA values, we will replace them with the median value of that column.
for(i in c('bc_util', 'mths_since_recent_bc', 'revol_util', 'percent_bc_gt_75', 'num_tl_120dpd_2m')) {
      loanData[i][is.na(loanData[i])] <- median(loanData[[i]], na.rm=TRUE)
}

# Checking again, all NAs are removed
naCounts <- sapply(loanData, function(x) sum(is.na(x)))
naCounts[naCounts >0]

# Now we will check for variabes that are highly correlated with eachother, finding 
# the names of those with a correlation higher than .8
corDf <- as.data.frame(cor(loanData[ sapply(loanData, is.numeric) ]))
highCorr <- unique(which(corDf > .8 & corDf < 1, arr.ind=TRUE)[,1])
corDf[highCorr, highCorr]

# Drop funded_amnt, funded_amnt_inv, fico_range_high, installment, tax_liens for 
# high correlation. The two funded categories are essentially the sam as loan amount, 
# and fico_high is essentially the same as fico_low so we can omit these variables. 
# Installment is basically calculated from loan_amnt, and tax_liens are likely contained
# in pub_rec, so we will omit these two.
corrDrops <- c('fico_range_high', 'funded_amnt', 'funded_amnt_inv', 'installment', 'tax_liens')
loanData <- loanData[, !(names(loanData) %in% corrDrops)]

# These dates are stored currently as factors. We could convert them to a date-time
# object, but I don't see how something like next_pymnt_d could be useful. I don't
# have a good strategy for transforming these dates into useful numbers, so I will
# opt not to use them. This is an area that could be given more thought.
dateDrops <- c('last_pymnt_d', 'next_pymnt_d', 'earliest_cr_line', 'last_credit_pull_d', 'issue_d')
loanData <- loanData[, !(names(loanData) %in% dateDrops)]

# Converting emp_length to numeric. 10+ -> 10. n/a and <1 -> 0. Assuming n/a is unemployed.
numericEmp <- gsub('\\+? years?', '', loanData$emp_length)
numericEmp <- gsub('(< 1|n/a)', '0', numericEmp)
numericEmp <- as.numeric(numericEmp)
loanData$emp_length <- numericEmp

# Fico scores do not make sense as numeric values. We will convert them to ranges.
# Function turns number into range label, based on ranges from Experian
ficoLabel <- function(x) {
      if(x < 580) 'Poor'
      else if(x < 670) 'Fair'
      else if(x < 740) 'Good'
      else if(x < 800) 'Very Good'
      else 'Excellent'
}
ficos <- sapply(loanData$fico_range_low, ficoLabel)
loanData$fico_range_low <- factor(ficos, levels = c('Poor', 'Fair', 'Good', 'Very Good', 'Excellent'))

# Looking at the number of each loan status type we have
summary(loanData$loan_status)
# In Grace Period is not relevant to our investigation. We should remove these, as they are neither delinquent nor current
loanData <- loanData[ loanData$loan_status != 'In Grace Period',]
# There are relatively few delinquent type loans, so it will likely be hard to distinguish between the 
# varietyies of delinquent loans. We will combine the types into one new category, Delinquent. We
# will also combine Current and Fully Paid into Current, essentially giving us "Good"
# and "Bad" categories. I'm not sure about Fully Paid. If we had more data, we could possibly
# leave the various "Bad" categories separate.
newStatus <- function(x) {
      if(x %in% c('Current', 'Fully Paid')) 'Current'
      else 'Delinquent'
}
status <- as.factor(sapply(loanData$loan_status, newStatus))
loanData$loan_status <- status

# Saving cleaned data
# write.csv(loanData, 'cleanedData.csv')

# Graphing: We'll take a look at histograms of some of our data, coloring by
# the loan status. This can give us some intuition as to how the data is
# split up by groups. These are just two examples, using fico score and
# loan amount, two things we might expect to be related to delinquency.
gfico <- ggplot(data = loanData, aes(x=fico_range_low, colour=loan_status)) +
      geom_bar()
gloan <- ggplot(data = loanData, aes(x=loan_status, y=loan_amnt)) +
      geom_boxplot(aes(colour=loan_status, fill=loan_status), alpha=.5) + 
      labs(x='Fico Score / Loan Status', y='Loan Amount',
                        title='Loan Amount Distribution by Fico Score') +
      facet_wrap(~fico_range_low, nrow=1) + 
      scale_colour_manual('Loan Status',values=c('Current'='limegreen','Delinquent'='red3', name='Loan Status')) +
      scale_fill_manual('Loan Status', values=c('Current'='limegreen', 'Delinquent'='red3', name='Loan Status'))
#gfico
gloan
ggsave('DsPlot.jpg', plot=gloan, width=8, height=6, units='in', dpi=200)
gamnt <- ggplot(data=loanData, aes(x=loan_amnt, colour=loan_status)) +
      geom_bar()
gamnt
ggsave('DsSized.jpg', plot=gamnt, width=6*3315/2589, heigh=6, unit='in', dpi=2589/6)# , scale=4032/1600)
# --------------------------------------
# MODEL BUILDING
# --------------------------------------

# The two options I am considering are a logistic regression model and a random
# forest model. Both these are good at classification, which is what we want to do.
# I will first try a logistic regression model, as the results are generally easier
# to interpret. Since we are interested in what factors might effect delinquency, 
# and not just predictive accuracy, I believe ease of interpretation is important.

# Our first step is to split our data into a training set and a test set with an
# 80/20 split. createDataPartition from the caret package will keep the same 
# proportion of our factor variable loan_status in each set.

set.seed(1121) # Set seed so results can be reproduced
trainIndex <- createDataPartition(loanData$loan_status, p = .8, list=FALSE)
train <- loanData[trainIndex,]
test <- loanData[-trainIndex,]
# This is needed for model.matrix, makes a formula of the form "~ var1+var2+....var_last"
myformula <- as.formula(paste('~ ', paste(names(select(train, -c(loan_status, id))), collapse='+')))
# We'll also split into predictors and outcomes x,y. We don't want to include id for prediction,
# but I wanted to keep it in the data so that I could look up specific entries if desired
trainx <- model.matrix(myformula,select(train, -c(loan_status, id)))
trainy <- train$loan_status
testx <- model.matrix(myformula, select(test, -c(loan_status, id)))
testy <- test$loan_status

# We have 35 possible predictor variables, which is a lot. We will use a Lasso
# regression to allow the model to find the most useful predictors. We will
# also use 10-fold cross validation to find the best value of the lambda
# parameter. Our data must be standardized for lasso, and alpha = 1
# is lasso, while alpha = 0 is ridge regression.
logModel <- cv.glmnet(trainx, trainy, family='binomial', standardize=TRUE, nfolds=10, alpha=1, type.measure = 'class')
# Plotting the model's classification error vs. the error if we predict all Current
baseError <- sum(trainy=='Delinquent')/length(trainy)
plot(logModel)
abline(h=baseError, lwd=3, col='blue')
# Predicting on test set
preds <- predict(logModel, s='lambda.min', newx =testx, type='class')
table(preds, testy)
preds[preds == 'Delinquent'] #No predictions were Delinquent, clearly we have a problem

# Our model is just pushing everything to 'Current' because te data is 90% Current
# We can try downsampling the Current data points to match the number of Delinquent
set.seed(5432)
downSampleCurrent <- sample_n( filter(train, loan_status=='Current'), summary(train$loan_status)[2]*2) #439 for theirs , summary(train$loan_status)
downTrain <- rbind(downSampleCurrent, filter(train, loan_status=='Delinquent'))
# upsample as well
downTrain <- rbind(downTrain, filter(train, loan_status=='Delinquent'))
downTrainx <- model.matrix(myformula, select(downTrain, -c(loan_status, id)))
downTrainy <- downTrain$loan_status

# This model has much lower accuracy, but the accuracy is roughly the same for both cases
logModelDown <- cv.glmnet(downTrainx, downTrainy, family='binomial', standardize=TRUE, nfolds=10, alpha=1, type.measure='class')
plot(logModelDown)
predsDown <- predict(logModelDown, s='lambda.1se', newx=testx, type='class')
table(predsDown, testy)
# Storing the coefficients of this model
coefs <- coef(logModelDown, s='lambda.1se')
# Looking at the fitted values vs training loan status
fittedDown <- predict(logModelDown, s='lambda.1se', newx=downTrainx, type='class')
table(fittedDown, downTrainy)

# Random forest - doing a quick accuracy comparison
rfModel <- randomForest(x = trainx, y=trainy)
rfModel$confusion
# Accuracy on test set is basically the same for the full data set
predsRf <- predict(rfModel, newdata=testx)
table(predsRf, testy)
# Same as above, build it with the downsampled data
rfModelDown <- randomForest(x = downTrainx, y=downTrainy)
# rfModelDown$confusion
# Results are very similar
predsRfDown <- predict(rfModelDown, newdata=testx)
table(predsRfDown, testy)
# Checking to see which predictions were the same in both
sum(predsRfDown == predsDown)/length(predsDown)
# Only 85% are the same, an ensemble method might be useful.

# Storing non-zero coefficients and names. No easy way to get this
# directly from glmnet
coefNames <- data.frame(names = row.names(coefs), coef = 1:length(coefs))
for(i in 1:42) {
      coefNames[i,2] <- coefs[i]} 
coefNames <- coefNames[coefNames$coef != 0,]
coefNames            

newFit <- glm(loan_status ~ loan_amnt + term + int_rate + verification_status + inq_last_6mths + last_pymnt_amnt + acc_open_past_24mths + 
                    mths_since_recent_bc + num_il_tl + pub_rec_bankruptcies, data=downTrain, family='binomial')
news <- predict(newFit, newdata=test, type='response')
newPreds <- ifelse(news > 0.5, 1,0)

newFit2 <- glm(loan_status ~ loan_amnt + term + emp_length + fico_range_low + int_rate + verification_status + inq_last_6mths + last_pymnt_amnt + acc_open_past_24mths + 
                     mths_since_recent_bc + num_il_tl + pub_rec_bankruptcies, data=downTrain, family='binomial')

resids <- residuals(newFit,type= 'pearson')
