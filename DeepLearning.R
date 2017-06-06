
rm(list=ls(all=TRUE))

library(RCurl)
data=read.table(text = getURL("https://raw.githubusercontent.com/rajsiddarth119/Datasets/master/Bank_dataset.csv"), header=T, sep=',',
                col.names = c('ID', 'age', 'exp', 'inc', 
                              'zip', 'family', 'ccavg', 'edu', 
                              'mortgage', 'loan', 'securities', 
                              'cd', 'online', 'cc'))
#Removing ID,Zipcode

data=subset(data,select = -c(ID,zip)) 

str(data)
#Numeric attributes : age,exp,inc,family,CCAvg,Mortgage
#Categorical: Education,Securities account,CD Account,Online,Credit card
#Target Variable: Personal Loan

num_data=data.frame(sapply(data[c('age','exp','inc','family','ccavg')],function(x){as.numeric(x)}))

#Categorical 
categ_attributes=c('edu','securities','cd','online')
categ_data=data.frame(sapply(data[categ_attributes],function(x){as.factor(x)}))
loan=as.factor(data$loan)

#Final data
data=cbind(num_data,categ_data,loan)

set.seed(1234) 

#Stratified sampling based on target variable
#install.packages("caTools")

str(data)
library(caTools)
index=sample.split(data$loan,SplitRatio = 0.7)
train_Data=data[index,]
test_Data=data[!index,]

#Building Random Forest model

#install.packages("randomForest")
library(randomForest)

model=randomForest(formula=loan~.,data = train_Data,ntrees=50,keep.forest=T,type="classification")

#Plotting variable importance plot
varImpPlot(model)
summary(model)

#Important eatures
var_imp=data.frame('attributes'=rownames(importance(model)),'gini index'=importance(model))
var_imp=var_imp[order(-var_imp$MeanDecreaseGini),]

#Classification using top 5 attributes
library(rpart)
var_imp$attributes=as.character(var_imp$attributes)

library(dplyr)
train_impvar=train_Data[c("loan",var_imp$attributes[1:5])]

#Building rpart model
model_impvar=rpart(loan~.,data = train_impvar,method = "class",cp=0.05)
#Plotting rpart model
plot(model_impvar)
text(model_impvar)
summary(model_impvar)

#Predicting using random forest model on test set

predict=predict(model_impvar,newdata = test_Data[c(var_imp$attributes[1:5])],type="class")
#Confusion matrix
conf_matrix=table(test_Data$loan,predict)
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
precision=conf_matrix[2,2]/sum(conf_matrix[,2])
print(paste("accuracy:",round(accuracy,2)))
print(paste("precision:",round(precision,2)))

###############################################################################################
#Building deep learning model
# Load H2o library
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "1g")

# Import a local R train data frame to the H2O cloud
train=as.h2o(x = train_Data, destination_frame = "train")
test=as.h2o(x=test_Data,destination_frame="test")

#Deep learning model
model_dl=h2o.deeplearning(x=names(train[-7]),autoencoder = T,training_frame = train,hidden = c(15),activation ="Tanh",epochs = 150 )

#Important features generated using deep learning
dl_features=as.data.frame(h2o.deepfeatures(model_dl,data = train[-7]))
dl_features_test=as.data.frame(h2o.deepfeatures(model_dl,data = test[-7]))


#Combining deep learning features with original features
train_dl=cbind(train_Data,dl_features)
test_dl=cbind(test_Data,dl_features_test)

#Applying random forests on model with original daa set and deep learning features
model=randomForest(loan~.,data = train_dl,keep.forest=T,ntree=30)
varImpPlot(model)  
summary(model)

#Predicting using random forest model on test set
predict=predict(model,newdata = test_dl,type="class")

#Confusion matrix
conf_matrix=table(test_dl$loan,predict)
accuracy=sum(diag(conf_matrix))/sum(conf_matrix)
precision=conf_matrix[2,2]/sum(conf_matrix[,2])
print(paste("accuracy:",round(accuracy,2)))
print(paste("precision:",round(precision,2)))

