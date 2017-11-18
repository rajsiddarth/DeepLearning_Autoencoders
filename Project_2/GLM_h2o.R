
rm(list=ls(all=TRUE))

library(RCurl)
data=read.table(text = getURL("https://raw.githubusercontent.com/rajsiddarth/Datasets/master/Bank_dataset.csv"), header=T, sep=',',
                col.names = c('ID', 'age', 'exp', 'inc', 
                              'zip', 'family', 'ccavg', 'edu', 
                              'mortgage', 'loan', 'securities', 
                              'cd', 'online', 'cc'))
#Removing ID,Zipcode

data=subset(data,select = -c(ID,zip)) 

str(data)
#Numeric attributes : age,exp,inc,family,CCAvg,
#Mortgage
#Categorical: Education,Securities account,CD Account,Online,
#Credit card
#Target Variable: Personal Loan

num_data=data.frame(sapply(data[c('age','exp','inc','family','ccavg')],function(x){as.numeric(x)}))

#Categorical to numerical
#install.packages("dummies")
library(dummies)
categ_attributes=c('edu','securities','cd','online')
categ_data=data.frame(sapply(data[categ_attributes],function(x){as.factor(x)}))
categ_data=dummy.data.frame(categ_data,sep="_")

loan=data$loan

#Final data
data=cbind(num_data,categ_data,loan)

set.seed(456) 

#Stratified sampling based on target variable
#install.packages("caTools")
library(caTools)
index=sample.split(data$loan,SplitRatio = 0.7)
train_Data=data[index,]
test_Data=data[!index,]

# Load H2o library
library(h2o)
h2o.removeAll()
# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "1g")

# Import a local R train data frame to the H2O cloud
train_data=as.h2o(x = train_Data, destination_frame = "train_data")

# Elastic net regularization with alpha=0.5 and classification with logistic regression
model = h2o.glm(y = "loan",x = setdiff(names(train_data),"loan"),training_frame = train_data, 
                   nfolds = 10,standardize = TRUE,seed = 123,alpha = 0.5,family = "binomial",lambda_search = TRUE)

print(model)
temp1=round(model@model$training_metrics@metrics$AUC,2)
temp2=round(model@model$cross_validation_metrics@metrics$AUC,2)
print(paste("training auc: ",temp1))
print(paste("cross-validation auc:",temp2))

# Using h2O glm grid search
lambda_opts = list(list(1), list(.5), list(.1), list(.01), 
                   list(.001), list(.0001), list(.00001), list(0))
alpha_opts = list(list(0), list(.25), list(.5), list(.75), list(1))

hyper_parameters = list(lambda = lambda_opts, alpha = alpha_opts)

# Build H2O GLM with grid search

grid_GLM=h2o.grid("glm",hyper_params = hyper_parameters,grid_id = "grid_GLM",
                     y = "loan",x = setdiff(names(train_data), "loan"),training_frame = train_data, 
                     family = "binomial")
summary(grid_GLM)

grid_GLM_models=lapply(grid_GLM@model_ids,function(model_id) { h2o.getModel(model_id) })

for (i in 1:length(grid_GLM_models)) 
{ 
  print(sprintf("regularization: %-50s auc: %f", grid_GLM_models[[i]]@model$model_summary$regularization, h2o.auc(grid_GLM_models[[i]])))
}

# Function to find the best model with respective to AUC
find_Best_Model=function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GLM_model = find_Best_Model(grid_GLM_models)

rm(grid_GLM_models)

# Get the auc of the best GBM model
best_GLM_model_AUC = h2o.auc(best_GLM_model)

# Examine the performance of the best model
best_GLM_model

# View the specified parameters of the best model
best_GLM_model@parameters

# Important Variables.
h2o.varimp(best_GLM_model)

# Import a local R test data frame to the H2O cloud
test_data= as.h2o(x = test_Data, destination_frame = "test_data")

# Predict on test data set
predict= h2o.predict(best_GLM_model, 
                          newdata = test_data[,setdiff(names(test_data), "loan")])

data_GLM = h2o.cbind(test_data[,"loan"], predict)

# Copy predictions from H2O to R
pred_GLM = as.data.frame(data_GLM)

# Shutdown H2O
h2o.shutdown(F)

# Hit Rate and Penetration calculation
conf_Matrix_GLM = table(pred_GLM$loan, pred_GLM$predict) 

Accuracy = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)
print(paste("Accuracy :",round(Accuracy,2)))


