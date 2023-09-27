
import numpy as np
import pandas as pd
import run_rf
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import model_training 
import modelo_opt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import math
import json
import pickle
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def feature_type(dataset):
    features=list(dataset.columns)
    unique_values={col: x[col].nunique() for col in dataset.columns}
    feat_type = ['CatBinary' if (x.name == 'category' and len(x.categories)==2) else'CatOrdinal' if x.name == 'category' else 'Numerical' for x in dataset.dtypes]
    features_type = dict(zip(features, feat_type))
    return features_type

def datos(dataset):
    if dataset=='boston':
        from sklearn.datasets import load_boston
        dataset=load_boston()
        dataset.keys()
        boston = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        min_max_scaler = preprocessing.MinMaxScaler()
        boston_s= min_max_scaler.fit_transform(boston)
        boston_s = pd.DataFrame(boston_s,columns=boston.columns)
        boston=boston_s
        boston["MEDV"] = dataset.target
        boston['target']=boston['MEDV'].apply(lambda x: -1 if x<=22 else 1) 
        boston=boston.drop('MEDV', axis=1) 
        boston['target']=boston['target'].astype('category', copy=False) 
        x=boston.drop('target',axis=1)
        x['CHAS']=x['CHAS'].astype('category')
        y=boston['target']
        var_related={}
        


    elif dataset=='compas':
        compas=pd.read_csv("compas_processed.csv",sep=";")
        compas['TwoYearRecid']=compas['TwoYearRecid'].apply(lambda x: 1 if x==0 else -1) #positive class 1 is no recid
        compas.loc[compas['AgeGroup'] == 'Less25', 'AgeGroup'] = 1
        compas.loc[compas['AgeGroup'] == '25-45', 'AgeGroup'] = 2
        compas.loc[compas['AgeGroup'] == 'More45', 'AgeGroup'] = 3

        to_onehot=['Race']
        df = compas
        df = df.dropna()
        transformer = make_column_transformer(
            (OneHotEncoder(), to_onehot),
            remainder='passthrough')
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())
        variables_onehot=[c for c in transformed_df.columns if 'onehot' in c]
        for col in variables_onehot:
            transformed_df[col]=transformed_df[col].astype('category')
        var_related={to_onehot[i]: [variables_onehot.index(char) for char in variables_onehot if 'x'+str(i) in char] for i in range(len(to_onehot))}
        compas2=transformed_df
        
        categ=['Sex','TwoYearRecid','ChargeDegree','PriorsCount','AgeGroup']
        for col in categ:
            compas2[col]=compas2[col].astype('category')

        x=compas2.drop(columns=['TwoYearRecid'])
        y=compas2['TwoYearRecid']

        

    elif dataset=="adult":

        adult=pd.read_csv("adult_mace2.csv",sep=",")
        adult['HoursPerWeek']=(adult['HoursPerWeek']-adult['HoursPerWeek'].min())/(adult['HoursPerWeek'].max()-adult['HoursPerWeek'].min())
        adult['CapitalLoss']=(adult['CapitalLoss']-adult['CapitalLoss'].min())/(adult['CapitalLoss'].max()-adult['CapitalLoss'].min())
        adult['CapitalGain']=(adult['CapitalGain']-adult['CapitalGain'].min())/(adult['CapitalGain'].max()-adult['CapitalGain'].min())
        adult['Age']=(adult['Age']-adult['Age'].min())/(adult['Age'].max()-adult['Age'].min())
        adult['Label']=adult['Label'].apply(lambda x: 1 if x==1 else -1)

        to_onehot=['WorkClass','MaritalStatus', 'Occupation', 'Relationship']
        df = adult
        df = df.dropna()
        transformer = make_column_transformer(
            (OneHotEncoder(), to_onehot),
            remainder='passthrough')
        transformed = transformer.fit_transform(df)
        transformed_df=pd.DataFrame.sparse.from_spmatrix(transformed,columns=transformer.get_feature_names()).sparse.to_dense()
        variables_onehot=[c for c in transformed_df.columns if 'onehot' in c]
        for col in variables_onehot:
            transformed_df[col]=transformed_df[col].astype('category')
        var_related={to_onehot[i]: [variables_onehot.index(char) for char in variables_onehot if 'x'+str(i) in char] for i in range(len(to_onehot))}
        adult2=transformed_df
        
        categ=['Sex','Label','EducationNumber','NativeCountry']
        
        for col in categ:
            adult2[col]=adult2[col].astype('category')


        x=adult2.drop(columns=['Label'])
        y=adult2['Label']

        

    elif dataset=="credit":
        credit=pd.read_csv("credit_processed.csv")
        

        credit['MaxBillAmountOverLast6Months']=(credit['MaxBillAmountOverLast6Months']-credit['MaxBillAmountOverLast6Months'].min())/(credit['MaxBillAmountOverLast6Months'].max()-credit['MaxBillAmountOverLast6Months'].min())
        credit['MaxPaymentAmountOverLast6Months']=(credit['MaxPaymentAmountOverLast6Months']-credit['MaxPaymentAmountOverLast6Months'].min())/(credit['MaxPaymentAmountOverLast6Months'].max()-credit['MaxPaymentAmountOverLast6Months'].min())
        credit['MostRecentBillAmount']=(credit['MostRecentBillAmount']-credit['MostRecentBillAmount'].min())/(credit['MostRecentBillAmount'].max()-credit['MostRecentBillAmount'].min())
        credit['MostRecentPaymentAmount']=(credit['MostRecentPaymentAmount']-credit['MostRecentPaymentAmount'].min())/(credit['MostRecentPaymentAmount'].max()-credit['MostRecentPaymentAmount'].min())

        

        categ=['NoDefaultNextMonth', 'isMale', 'isMarried', 'AgeGroup',
       'EducationLevel',
       'MonthsWithZeroBalanceOverLast6Months',
       'MonthsWithLowSpendingOverLast6Months',
       'MonthsWithHighSpendingOverLast6Months', 'TotalOverdueCounts', 'TotalMonthsOverdue',
       'HasHistoryOfOverduePayments']

        for col in categ:
            credit[col]=credit[col].astype('category')

        var_related={}

        x=credit.drop(columns=["NoDefaultNextMonth"])
        y=credit["NoDefaultNextMonth"]
        y=pd.Series(y).apply(lambda x: 1 if x==1 else -1).astype('category')

        

    elif dataset=="Students":
        students=pd.read_csv("Students-Performance-MAT.csv")
        students['Class']=students['Class'].apply(lambda x: 1 if x==1 else -1)
      
        #binary variables to 0-1
        students['school']=students['school'].apply(lambda x: 1 if x=='MS' else 0)
        students['sex']=students['sex'].apply(lambda x: 1 if x=='F' else 0)
        students['address']=students['address'].apply(lambda x: 1 if x=='U' else 0)
        students['famsize']=students['famsize'].apply(lambda x: 1 if x=='GT3' else 0)
        students['Pstatus']=students['Pstatus'].apply(lambda x: 1 if x=='A' else 0)
        students['schoolsup']=students['schoolsup'].apply(lambda x: 1 if 'yes' else 0)
        students['famsup']=students['famsup'].apply(lambda x: 1 if 'yes' else 0)
        students['paid']=students['paid'].apply(lambda x: 1 if 'yes' else 0)
        students['activities']=students['activities'].apply(lambda x: 1 if 'yes' else 0)
        students['nursery']=students['nursery'].apply(lambda x: 1 if 'yes' else 0)
        students['higher']=students['higher'].apply(lambda x: 1 if 'yes' else 0)
        students['internet']=students['internet'].apply(lambda x: 1 if 'yes' else 0)
        students['romantic']=students['romantic'].apply(lambda x: 1 if 'yes' else 0)

        

        to_onehot=["Mjob","Fjob","reason","guardian"]

        df = students
        df = df.dropna()
        transformer = make_column_transformer(
            (OneHotEncoder(), to_onehot),
            remainder='passthrough')
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())
        variables_onehot=[c for c in transformed_df.columns if 'onehot' in c]
        for col in variables_onehot:
            transformed_df[col]=transformed_df[col].astype('category')
        var_related={to_onehot[i]: [variables_onehot.index(char) for char in variables_onehot if 'x'+str(i) in char] for i in range(len(to_onehot))}
        students2=transformed_df
        


        categ=["age","school","sex","address","famsize","Pstatus","Medu","Fedu","traveltime","studytime","failures",
             "schoolsup","famsup","paid","activities","nursery","higher","internet","romantic","famrel","freetime","goout",
             "Dalc","Walc","health","absences"]


        for col in categ:
            students2[col]=students2[col].astype('category')

        x=students2.drop(columns=['Class'])
        y=students2['Class']

    return x, y, var_related



##choose data: boston, compas, Students, credit
x,y, var_related=datos("compas")
features=x.columns
features_type=feature_type(x)
index_cat_bin=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='CatBinary']
index_cat_ord=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='CatOrdinal']
index_cont=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='Numerical']
lower_ord={}
upper_ord={}
for i in index_cat_ord:
    lower_ord[i]=min(x[features[i]])
    upper_ord[i]=max(x[features[i]])
weight_discrete={}
for i in index_cat_ord: 
    weight_discrete[i]=len(x[features[i]].unique()) #weight binary: 0.5



#create a classification model
#RF
leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical_bin,constraints_left_categorical_bin,constraints_right_categorical_ord,constraints_left_categorical_ord, index, x, y, y_pred, y_pred_prob, model_clas, tree_data=model_training.randomforest(100,3,x,y,features_type)


#SVM or LR
x,y,y_pred,y_pred_prob,model,w, b=model_training.linear_model('LR',x,y) 




#create the optimization model

#rf separable
objective='l0l2'
model_opt=modelo_opt.modelo_opt_rf_separable(leaves,index_cont,index_cat_bin,index_cat_ord,var_related,objective)

#rf collective
objective="l2l0global"
model_opt=modelo_opt.modelo_opt_rf_nonseparable(leaves,index_cont,index_cat_bin, index_cat_ord,var_related,objective)

#lineal collective
objective='l2l0global'
model_opt=modelo_opt.modelo_opt_lineal_nonseparable(index_cont,index_cat_bin, index_cat_ord,var_related,objective,"False")

objective='l2'
model_opt=modelo_opt.modelo_opt_lineal_nonseparable(index_cont,index_cat_bin, index_cat_ord,var_related,objective,"True")

#lineal separable
objective='l0l2'
model_opt=modelo_opt.modelo_opt_lineal_separable(index_cont,index_cat_bin, index_cat_ord,var_related,objective)


##choose x0
x0=x[y_pred.squeeze()==-1]
y0=y_pred[y_pred.squeeze()==-1].squeeze()

#compas
indices=[1603,4253,6022,1446,1969,3612,3619,5829,4302,6160]
x0=x.loc[indices]
y0=y_pred.loc[indices].squeeze()


#credit
indices=[1107, 1860, 3160, 4278, 6810, 7177, 14443, 16500, 24211, 26881]
x0=x.loc[indices]
y0=y_pred.loc[indices].squeeze()

#boston
indices=[9, 49, 60, 154, 312, 373, 386, 398, 426, 496]
x0=x.loc[indices]
y0=y_pred.loc[indices].squeeze()

#students
indices=[11, 39, 44, 124, 175, 201, 225, 246, 300, 349]
x0=x.loc[indices]
y0=y_pred.loc[indices].squeeze()



indices=x0.index

#perc=round(0.95*x0.shape[0]) #percentage change
perc=x0.shape[0]
nu=0.50 #prob of belonging to the positive class
lam=0.2 #lam multiplying l0
sol_ini=[]
timelimit=3200

#solve rf separable
result={}
final_class={}
x_sols={}
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_individual(i,timelimit,x0,y0,lower_ord,upper_ord,weight_discrete,model_opt,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical_bin,constraints_left_categorical_bin,constraints_right_categorical_ord,constraints_left_categorical_ord,features_type,index_cont,index_cat_bin,index_cat_ord,model_clas,tree_data,lam,nu,x,y_pred)
cambio=pd.DataFrame(result,index=features).transpose()


#solve lineal separable
result={}
final_class={}
x_sols={}
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_lineal_ind(i,x0,y0,lower_ord,upper_ord,weight_discrete,model_opt,w,b,index_cont,index_cat_bin,index_cat_ord,features_type,lam,nu)
cambio=pd.DataFrame(result,index=features).transpose()



#solve lineal non-separable
x_sol,cambio=run_rf.optimization_lineal_collective(x0,y0,lower_ord,upper_ord,weight_discrete,perc,model_opt,w,b,index_cont,index_cat_bin,index_cat_ord,features_type,lam,nu,20,timelimit)

#solve rf non-separable
x_ini=[]
x_sol,cambio=run_rf.optimization_collective(x0,y0,lower_ord,upper_ord,weight_discrete,perc,model_opt,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical_bin,constraints_left_categorical_bin,constraints_right_categorical_ord,constraints_left_categorical_ord,features_type,index_cont,index_cat_bin,index_cat_ord,model_clas,tree_data,timelimit,x_ini,lam,nu)


#to visualize the heatmaps better (for the one-hot encoding variables)
cambio.columns = [x.split('_')[3] if 'onehot' in x else x for x in list(cambio.columns)]



#heatmaps
plt.clf()
sns.set(rc={'figure.figsize':(20, 6)})
#sns.set(rc={'figure.figsize':(8, 20)})
#data_sinage=data.drop(columns=['age_factor','priors_count'])
ax = sns.heatmap(cambio,linewidths=.5,center=0, cmap="PiYG",yticklabels=False,square=True)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=40, ha="left")
plt.yticks(rotation=0)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
#plt.savefig('RF_ind002_insrandom_credit_depth4.png')
plt.tight_layout()
plt.show()





#heatmap at different scales

#for credit
# adapt the colormaps such that the "under" or "over" color is "none"
cmap1 = plt.get_cmap('PiYG') #'crest' #vlag
cmap1.set_under('none')
cmap2 = plt.get_cmap('PiYG')
cmap2.set_over('none')
cmap3= plt.get_cmap('vlag')
cmap3.set_under('none')
cmap3.set_over('none')
pal = sns.diverging_palette(260, 30, s=100, center='light', as_cmap=True)
cmap4=plt.get_cmap(pal)
cmap4.set_under('none')
cmap4.set_over('none')

sns.set(rc={'figure.figsize':(20, 10)})
ax1 = sns.heatmap(cambio, linewidths=.5, vmin=0.5, vmax=1, center=0, cmap=cmap1, cbar_kws={'pad': -0.04})
ax2= sns.heatmap(cambio,linewidth=.5, vmin=-0.2,vmax=0.2, cmap=cmap4, ax=ax1, cbar_kws={'pad':-0.02})
ax3=sns.heatmap(cambio,linewidths=.5, vmax=-0.5, vmin=-8, center=0, cmap=cmap2, ax=ax1,yticklabels=False,square=True)
ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=7, rotation=40, ha="left")
plt.yticks(rotation=0)
ax3.xaxis.tick_top() # x axis on top
ax3.xaxis.set_label_position('top')
#plt.tight_layout()
#plt.savefig('LR_i02_insrandom_credit.png',bbox_inches='tight')
plt.show()

#compas and students
cmap1 = plt.get_cmap('PiYG') #'crest' #vlag
cmap1.set_under('none')
cmap2 = plt.get_cmap('PiYG')
cmap2.set_over('none')
cmap3= plt.get_cmap('vlag')
cmap3.set_under('none')
cmap3.set_over('none')
pal = sns.diverging_palette(260, 30, s=100, center='light', as_cmap=True)
cmap4=plt.get_cmap(pal)
cmap4.set_under('none')
cmap4.set_over('none')


#for compas and students
#sns.set(rc={'figure.figsize':(30, 6)})
sns.set(rc={'figure.figsize':(10, 6)})
ax1 = sns.heatmap(cambio, linewidths=.5, vmin=0, vmax=1, center=0, cmap=cmap1, cbar_kws={'pad': -0.03,"shrink": 1})
ax2= sns.heatmap(cambio,linewidth=.5, vmin=-2,vmax=-1, cmap=cmap4, ax=ax1, cbar_kws={'pad':-0.02,"shrink": 1})
ax3=sns.heatmap(cambio,linewidths=.5, vmax=-4, vmin=-37, center=0, cmap=cmap2, ax=ax1,yticklabels=False,square=True,cbar_kws={"shrink":1})
ax3.set_xticklabels(ax2.get_xticklabels(), fontsize=7, rotation=40, ha="left")
plt.yticks(rotation=0)
ax3.xaxis.tick_top() # x axis on top
ax3.xaxis.set_label_position('top')
#plt.tight_layout()
plt.savefig('RF_glob02_insrandom_compas.png',bbox_inches='tight')
plt.show()












