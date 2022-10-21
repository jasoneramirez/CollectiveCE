
import numpy as np
import pandas as pd
import run_rf
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import model_training 
import modelo_opt
from sklearn import preprocessing




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
leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical, index, x, y, y_pred, y_pred_prob, model_clas, tree_data=model_training.randomforest(13,3,x,y)

#SVM or LR
x,y,y_pred,model,w, b=model_training.linear_model('LR',x,y)


#to reproduce the experiments of the paper

import pickle
with open('rf_norm_100_prune3_todos.pkl', 'rb') as f: 
        leaves, values, restricciones_right_numerical, restricciones_left_numerical, restricciones_right_categorical,restricciones_left_categorical, index, x2, y, y_pred, y_pred_prob,model_clas,tree_data =pickle.load(f)

constraints_right_categorical=[(a,b,3,d) for (a,b,c,d) in restricciones_right_categorical]
constraints_left_categorical=[(a,b,3,d) for (a,b,c,d) in restricciones_left_categorical]
constraints_right_numerical=[(a,b,c,d) if c<=2 else (a,b,c+1,d) for (a,b,c,d) in restricciones_right_numerical]
constraints_left_numerical=[(a,b,c,d) if c<=2 else (a,b,c+1,d) for (a,b,c,d) in restricciones_left_numerical]



#create the optimization model
objective='l0l2'
model_opt=modelo_opt.modelo_opt_rf_separable(leaves,index_cont,index_cat,objective)

#collective
model_opt_col=modelo_opt.modelo_opt_rf_nonseparable(leaves,index_cont,index_cat)

#lineal
objective='l0l2'
model_opt=modelo_opt.modelo_opt_lineal_separable(index_cont,index_cat,objective)


#define x0 and solve

index_criminalidad_alta=x[x.CRIM>0.002812].index.tolist()
x_crim=x.loc[index_criminalidad_alta]
y_crim=y_pred.loc[index_criminalidad_alta]

x_pos=x_crim[y_crim.squeeze()==1][0:10].index.tolist()
x_neg=x_crim[y_crim.squeeze()==-1][0:10].index.tolist()

#separable
x0=x.iloc[x_pos+x_neg]
y0=y_pred.iloc[x_pos+x_neg].squeeze()
indices=x0.index
lam=0.01 #lambda*l0+l2

#noseparable
x0=x.iloc[x_neg]
y0=y_pred.iloc[x_neg].squeeze()
indices=x0.index
perc=10 #number of people to be changed

#define timelimit
timelimit=100


result={}
final_class={}
x_sols={}
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_individual(i,timelimit,x0,y0,lam,model_opt,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,x,y)


#lineal
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_lineal_ind(i,x0,y0,lam,model_opt,w,b,index_cont,index_cat)



#no-separable

import pickle
with open('sol_ini_c.pkl', 'rb') as f:
        sol_ini=pickle.load(f)

data=run_rf.optimization_collective(x0,y0,perc,model_opt_col,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,timelimit,sol_ini)



# the heatmaps

def data_heatmap(resul,final_class,features,indices):

   

    resul_neg={} #ind que pasan de neg a pos
    resul_pos={} #ind que pasan de pos a neg


    for i in indices:
        if final_class[i]==1:
            resul_neg[i]=resul[i]
        elif final_class[i]==-1:
            resul_pos[i]=resul[i]


    data=pd.DataFrame(resul,index=features)

    data1=pd.DataFrame(resul_neg,index=features).transpose()
    data2=pd.DataFrame(resul_pos,index=features).transpose()


    return data1,data2

data1_p,data2_p=data_heatmap(result,final_class,features,indices)

plt.clf()
sns.set(rc={'figure.figsize':(12, 8)})
ax = sns.heatmap(data2_p,linewidths=.5,center=0,vmin=-0.3,vmax=0.3, cmap="PiYG",yticklabels=False,square=True)
plt.yticks(rotation=0)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
#plt.savefig('posaneg_rf100_l2.png')
plt.show()




