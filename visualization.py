
import numpy as np
import pandas as pd
import run_rf
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import model_training 
import modelo_opt
from sklearn import preprocessing
from numpy import linalg as l2
import json
import pickle
import math



#example

#data and data characteristics

def feature_type(dataset):
        features=list(dataset.columns)
        feat_type = ['Categorical' if x.name == 'category' else 'Numerical' for x in dataset.dtypes]
        features_type = dict(zip(features, feat_type))
        return features_type

def datos():
    from sklearn.datasets import load_boston
    dataset=load_boston()
    dataset.keys()
    boston = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    min_max_scaler = preprocessing.MinMaxScaler()
    boston_s= min_max_scaler.fit_transform(boston)
    boston_s = pd.DataFrame(boston_s,columns=boston.columns)
    boston=boston_s
    boston["MEDV"] = dataset.target
    boston['target']=boston['MEDV'].apply(lambda x: -1 if x<=22 else 1) #lo paso a clasificacion
    boston=boston.drop('MEDV', axis=1) 
    boston['target']=boston['target'].astype('category', copy=False) 
    x=boston.drop('target',axis=1)
    x['CHAS']=x['CHAS'].astype('category') #no olvidar indicar cuales son categoricas
    y=boston['target']
    return x, y

x,y=datos()
features=x.columns
features_type=feature_type(x)
index_cat=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='Categorical']
index_cont=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='Numerical']

#create a classification model

#RF
leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical, index, x, y, y_pred, y_pred_prob, model_clas, tree_data=model_training.randomforest(100,3,x,y)

#SVM or LR
x,y,y_pred,model,w, b=model_training.linear_model('LR',x,y)



#create the optimization model
objective='l0l2'
model_opt=modelo_opt.modelo_opt_rf_separable(leaves,index_cont,index_cat,objective)

#collective
objective="l2l0global"
model_opt_col=modelo_opt.modelo_opt_rf_nonseparable(leaves,index_cont,index_cat,objective,'False')


#lineal
objective='l0l2'
model_opt=modelo_opt.modelo_opt_lineal_separable(index_cont,index_cat,objective)

#linealcollective
objective="l2l0global"
model_opt_col=modelo_opt.modelo_opt_lineal_nonseparable(index_cont,index_cat,objective,'False')




#define x0 and solve


x0=x[y_pred.squeeze()==-1]
y0=y_pred[y_pred.squeeze()==-1].squeeze()


#10 instances
indices=[9, 49, 60, 154, 312, 373, 386, 398, 426, 496]
x0=x.loc[indices]
y0=y_pred.loc[indices].squeeze()

indices=x0.index

#indicate percentage
perc=round(0.90*x0.shape[0],0)


nu=0.5
lam=0.02
maxf=1



#solve separable case Random Forest
result={}
final_class={}
x_sols={}
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_individual(i,x0,y0,lam,model_opt,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,x,y,nu)
data=pd.DataFrame(result,index=features).transpose()

#solve separable case Linear Classifier
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_lineal_ind(i,x0,y0,lam,model_opt,w,b,index_cont,index_cat)
#data solution separable
data=pd.DataFrame(result,index=features).transpose()

#no-separable

#import pickle
#with open('sol_ini_c.pkl', 'rb') as f:
#        sol_ini=pickle.load(f)

sol_ini={}

#solve non-separable Random Forest
data,fobj=run_rf.optimization_collective(x0,y0,perc,model_opt_col,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,sol_ini,lam,nu)

#solve non-separable Linear Classifer
data,fobj=run_rf.optimization_lineal_collective(x0,y0,perc,model_opt_col,w,b,index_cont,index_cat,lam,nu,maxf)





