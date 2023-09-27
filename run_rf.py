# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:03:17 2020

@author: Jasone
"""

# el run

from __future__ import division
#import Pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from numpy import linalg as l2

import modelo_opt 
import model_training 
import numpy as np
import pandas as pd
import os
import math



def optimization_collective(x0,y0,lowerOrd,upperOrd,weight_discrete,perc,model,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical_bin,constraints_left_categorical_bin,constraints_right_categorical_ord,constraints_left_categorical_ord,features_type,index_cont,index_cat_bin,index_cat_ord,model_clas,tree_data,timelimit,x_ini,lam,nu):
           
    n_features=x0.shape[1]
    n_ind=x0.shape[0]
    n_trees=len(leaves)
    features=list(features_type.keys())
    #features_type=feature_type(x0)
    indices=list(x0.index)
    nu2=n_trees*(2*nu-1)

    x0_1={}
    x0_2a={}
    x0_2b={}

    cat_bin_features=[f for f in features if features_type[f]=='CatBinary']
    cat_ord_features=[f for f in features if features_type[f]=='CatOrdinal']
    cont_features=[f for f in features if features_type[f]=='Numerical']
    n_cat_bin=len(cat_bin_features)
    n_cat_ord=len(cat_ord_features)
    n_cont=len(cont_features)

    for j in range(0,n_ind):
        for (f,i) in zip(cont_features,index_cont):
            x0_1[i,j]=x0.iloc[j][f]

    for j in range(0,n_ind):
        for (f,i) in zip(cat_bin_features,index_cat_bin):
            x0_2a[i,j]=x0.iloc[j][f]
    
    for j in range(0,n_ind):
        for (f,i) in zip(cat_ord_features,index_cat_ord):
            x0_2b[i,j]=x0.iloc[j][f]

    y_final={}
    for j in range(0,n_ind):
        y_final[j]=-list(y0)[j]

    #initial solution:
    
    if isinstance(x_ini,pd.DataFrame):
        x_ini_1={}
        for j in range(0,n_ind):
            for (f,i) in zip(cont_features,index_cont):
                x_ini_1[i,j]=x_ini.iloc[j][f]

        x_ini_2a={}   
        for j in range(0,n_ind):
            for (f,i) in zip(cat_bin_features,index_cat_bin):
                x_ini_2a[i,j]=x_ini.iloc[j][f]

        x_ini_2b={}   
        for j in range(0,n_ind):
            for (f,i) in zip(cat_ord_features,index_cat_ord):
                x_ini_2b[i,j]=x_ini.iloc[j][f]

        sol_ini=model_clas.apply(x_ini)  #necesito el modelo
        z_sol_ini={}
        for i in range(0,n_ind):
            z_sol_ini[i]=list(map(lambda tl:tree_data[tl[0]]['index_leaves'].index(tl[1]),enumerate(sol_ini[i]))) #necesito datos_arbol


        z={}
        for i in range(0,n_ind):
            for t in range(n_trees):
                for l in range(leaves[t]):
                    z[t,l,i]=0
                    z[t,z_sol_ini[i][t],i]=1

        D={}
        for i in range(0,n_ind):
            for t in range(n_trees):
                D[t,i]=0
                for l in range(leaves[t]):
                    D[t,i]=D[t,i]+values[t][l]*z[t,l,i]



    leaf={}
    for t in range(n_trees):
        leaf[t]=leaves[t]


    n_left_num=len(constraints_left_numerical)
    n_right_num=len(constraints_right_numerical)
    n_left_cat_bin=len(constraints_left_categorical_bin)
    n_right_cat_bin=len(constraints_right_categorical_bin)
    n_left_cat_ord=len(constraints_left_categorical_ord)
    n_right_cat_ord=len(constraints_right_categorical_ord)

    restric_left_num={}
    for i in range(n_left_num):
        restric_left_num[i]=constraints_left_numerical[i]

    restric_right_num={}
    for i in range(n_right_num):
        restric_right_num[i]=constraints_right_numerical[i]

    restric_left_cat_bin={}
    for i in range(n_left_cat_bin):
        restric_left_cat_bin[i]=constraints_left_categorical_bin[i]

    restric_right_cat_bin={}
    for i in range(n_right_cat_bin):
        restric_right_cat_bin[i]=constraints_right_categorical_bin[i]

    restric_left_cat_ord={}
    for i in range(n_left_cat_ord):
        restric_left_cat_ord[i]=constraints_left_categorical_ord[i]

    restric_right_cat_ord={}
    for i in range(n_right_cat_ord):
        restric_right_cat_ord[i]=constraints_right_categorical_ord[i]



    values_leaf_dict={}
    for t in range(n_trees):
        for l in range(leaves[t]):
            values_leaf_dict[(t,l)]=values[t][l]

    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2a = {None : n_cat_bin},#n variables categoricas
            N2b= {None: n_cat_ord},
            #x0_0={None:0},
            ind ={None: n_ind},
            M1={None:100}, 
            M2={None:100}, 
            M3={None:100}, 
            epsi={None:1e-6},
            trees = {None:n_trees},
            leaves = leaf,
            values_leaf=values_leaf_dict,
            nleft_num={None:n_left_num},
            nright_num={None:n_right_num},
            nleft_cat_bin={None:n_left_cat_bin},
            nright_cat_bin={None:n_right_cat_bin},
            nleft_cat_ord={None:n_left_cat_ord},
            nright_cat_ord={None:n_right_cat_ord},
            left_num=restric_left_num,
            right_num=restric_right_num,
            left_cat_bin=restric_left_cat_bin,
            right_cat_bin=restric_right_cat_bin,
            left_cat_ord=restric_left_cat_ord,
            right_cat_ord=restric_right_cat_ord,
            x0_1=x0_1,
            x0_2a=x0_2a,
            x0_2b=x0_2b,
            lowerOrd=lowerOrd,
            upperOrd=upperOrd,
            weight_discrete=weight_discrete,
            y_f=y_final,
            perc={None: perc},
            lam={None:lam},
            nu={None:nu2}
            )}


  
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")
    opt.options['TimeLimit'] = timelimit

    if isinstance(x_ini,pd.DataFrame):

        for j in range(0,n_ind):
            for i in index_cont:
                instance.x_1[i,j]=x_ini_1[i,j]
    
        for j in range(0,n_ind):
            for i in index_cat_bin:
                instance.x_2a[i,j]=x_ini_2a[i,j]

        for j in range(0,n_ind):
            for i in index_cat_ord:
                instance.x_2b[i,j]=x_ini_2b[i,j]

        for i in range(0,n_ind):
            for t in range(n_trees):
                for l in range(leaves[t]):
                    instance.z[t,l,i]=z[t,l,i]

        for i in range(0,n_ind):
            for t in range(n_trees):
                instance.D[t,i]=D[t,i]
      
    results = opt.solve(instance,tee=True) # tee=True: ver iteraciones por pantalla




    x_sol_aux=np.zeros([len(features),n_ind])

    for i in index_cont:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_1[i,j].value
    for i in index_cat_bin:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_2a[i,j].value

    for i in index_cat_ord:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_2b[i,j].value
    
    

    z_sol={}
    for i in range(0,n_ind):
            for t in range(n_trees):
                for l in range(leaves[t]):
                    z_sol[t,l,i]=instance.z[t,l,i].value


    x_sol=pd.DataFrame(x_sol_aux,features)

    print(x_sol)




    cambio_x=np.zeros([len(features),n_ind])
    

    for i in range(len(features)):
        for j in range(n_ind):
            cambio_x[i,j]=x_sol_aux[i,j]-x0.iloc[j,i]
            if cambio_x[i,j]<=1e-10:
                cambio_x[i,j]==0
    

    data=pd.DataFrame(cambio_x,index=features)

    data=data.T



    return x_sol, data

def optimization_individual(i,timelimit,x0,y0,lowerOrd,upperOrd,weight_discrete,model,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical_bin,constraints_left_categorical_bin,constraints_right_categorical_ord,constraints_left_categorical_ord,features_type,index_cont,index_cat_bin,index_cat_ord,model_clas,tree_data,lam,nu,x,y_pred):
     
    
    n_features=x0.shape[1]
    n_ind=x0.shape[0]
    n_trees=len(leaves)
    features=list(features_type.keys())

    nu2=n_trees*(2*nu-1)

    x0=x0.loc[i]
    y0=y0[i]

    x0_1={}
    x0_2a={}
    x0_2b={}

    cat_bin_features=[f for f in features if features_type[f]=='CatBinary']
    cat_ord_features=[f for f in features if features_type[f]=='CatOrdinal']
    cont_features=[f for f in features if features_type[f]=='Numerical']
    n_cat_bin=len(cat_bin_features)
    n_cat_ord=len(cat_ord_features)
    n_cont=len(cont_features)

    for (f,i) in zip(cont_features,index_cont):
        x0_1[i]=x0[f]

    for (f,i) in zip(cat_bin_features,index_cat_bin):
        x0_2a[i]=x0[f]
    
    for (f,i) in zip(cat_ord_features,index_cat_ord):
        x0_2b[i]=x0[f]


    
    distance=100
    for i in range(x.shape[0]):
        if y_pred.iloc[i][0]==-y0: #y_pred
            d=l2.norm(x0-x.iloc[i])
            if d<=distance:
                distance=d
                x_ini=x.iloc[i] #initial solution
    

    x_ini_1={}
    x_ini_2a={}
    x_ini_2b={}

    for (f,i) in zip(cont_features,index_cont):
        x_ini_1[i]=x0[f]

    for (f,i) in zip(cat_bin_features,index_cat_bin):
        x_ini_2a[i]=x0[f]
    
    for (f,i) in zip(cat_ord_features,index_cat_ord):
        x_ini_2b[i]=x0[f]
    

    x_ini=np.array(x_ini).reshape(1,-1)
    sol_ini=model_clas.apply(x_ini)[0]  
    z_sol_ini=list(map(lambda tl:tree_data[tl[0]]['index_leaves'].index(tl[1]),enumerate(sol_ini))) #necesito datos_arbol

    z={}
    for t in range(n_trees):
        for l in range(leaves[t]):
            z[t,l]=0
            z[t,z_sol_ini[t]]=1

    D={}
    for t in range(n_trees):
        D[t]=0
        for l in range(leaves[t]):
            D[t]=D[t]+values[t][l]*z[t,l]

  

    leaf={}
    for t in range(n_trees):
        leaf[t]=leaves[t]


    n_left_num=len(constraints_left_numerical)
    n_right_num=len(constraints_right_numerical)
    n_left_cat_bin=len(constraints_left_categorical_bin)
    n_right_cat_bin=len(constraints_right_categorical_bin)
    n_left_cat_ord=len(constraints_left_categorical_ord)
    n_right_cat_ord=len(constraints_right_categorical_ord)

    restric_left_num={}
    for i in range(n_left_num):
        restric_left_num[i]=constraints_left_numerical[i]

    restric_right_num={}
    for i in range(n_right_num):
        restric_right_num[i]=constraints_right_numerical[i]

    restric_left_cat_bin={}
    for i in range(n_left_cat_bin):
        restric_left_cat_bin[i]=constraints_left_categorical_bin[i]

    restric_right_cat_bin={}
    for i in range(n_right_cat_bin):
        restric_right_cat_bin[i]=constraints_right_categorical_bin[i]

    restric_left_cat_ord={}
    for i in range(n_left_cat_ord):
        restric_left_cat_ord[i]=constraints_left_categorical_ord[i]

    restric_right_cat_ord={}
    for i in range(n_right_cat_ord):
        restric_right_cat_ord[i]=constraints_right_categorical_ord[i]



    values_leaf_dict={}
    for t in range(n_trees):
        for l in range(leaves[t]):
            values_leaf_dict[(t,l)]=values[t][l]
    data= {None: dict(
            N1 = {None : n_cont}, #n variables continuas
            N2a = {None : n_cat_bin},#n variables categoricas
            N2b= {None: n_cat_ord},
            M1={None:100}, 
            M2={None:100}, 
            M3={None:100}, 
            epsi={None:1e-6},
            trees = {None:n_trees},
            leaves = leaf,
            values_leaf=values_leaf_dict,
            nleft_num={None:n_left_num},
            nright_num={None:n_right_num},
            nleft_cat_bin={None:n_left_cat_bin},
            nright_cat_bin={None:n_right_cat_bin},
            nleft_cat_ord={None:n_left_cat_ord},
            nright_cat_ord={None:n_right_cat_ord},
            left_num=restric_left_num,
            right_num=restric_right_num,
            left_cat_bin=restric_left_cat_bin,
            right_cat_bin=restric_right_cat_bin,
            left_cat_ord=restric_left_cat_ord,
            right_cat_ord=restric_right_cat_ord,
            x0_1=x0_1,
            x0_2a=x0_2a,
            x0_2b=x0_2b,
            lowerOrd=lowerOrd,
            upperOrd=upperOrd,
            weight_discrete=weight_discrete,
            y={None:-y0},
            lam={None:lam},
            nu={None:nu2}
            )}


  
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")
    opt.options['TimeLimit'] = timelimit

    for i in index_cont:
        instance.x_1[i]=x_ini_1[i]

    for i in index_cat_bin:
        instance.x_2a[i]=x_ini_2a[i]

    for i in index_cat_ord:
        instance.x_2b[i]=x_ini_2b[i]

    for t in range(n_trees):
        for l in range(leaves[t]):
            instance.z[t,l]=z[t,l]

    for t in range(n_trees):
        instance.D[t]=D[t]

      
    results = opt.solve(instance,tee=True,warmstart=True) # tee=True: ver iteraciones por pantalla
   
 
    x_sol_aux=np.zeros(len(features))
    for i in index_cont:
        x_sol_aux[i]=instance.x_1[i].value
    for i in index_cat_bin:
        x_sol_aux[i]=instance.x_2a[i].value
    for i in index_cat_ord:
        x_sol_aux[i]=instance.x_2b[i].value
   



    x_sol=pd.DataFrame(x_sol_aux,features,columns=['x'])


    salida=-y0*sum([instance.D[t].value for t in range(n_trees)])
    print('salida: '+str(salida))


    cambio_x=np.zeros(len(features))

    for j in range(len(features)):
        cambio_x[j]=x_sol_aux[j]-x0[j]
    


 
    

    return x_sol, cambio_x, -y0



def optimization_lineal_collective(x0,y0,lowerOrd,upperOrd,weight_discrete,perc,model,w,b,index_cont,index_cat_bin,index_cat_ord,features_type,lam,nu,maxf,timelimit):
    
    n_features=x0.shape[1]
    n_ind=x0.shape[0]
    features=list(features_type.keys())
    indices=list(x0.index)



    x0_1={}
    x0_2a={}
    x0_2b={}

    cat_bin_features=[f for f in features if features_type[f]=='CatBinary']
    cat_ord_features=[f for f in features if features_type[f]=='CatOrdinal']
    cont_features=[f for f in features if features_type[f]=='Numerical']
    n_cat_bin=len(cat_bin_features)
    n_cat_ord=len(cat_ord_features)
    n_cont=len(cont_features)

    for j in range(0,n_ind):
        for (f,i) in zip(cont_features,index_cont):
            x0_1[i,j]=x0.iloc[j][f]

    for j in range(0,n_ind):
        for (f,i) in zip(cat_bin_features,index_cat_bin):
            x0_2a[i,j]=x0.iloc[j][f]
    
    for j in range(0,n_ind):
        for (f,i) in zip(cat_ord_features,index_cat_ord):
            x0_2b[i,j]=x0.iloc[j][f]

    y_final={}
    for j in range(0,n_ind):
        y_final[j]=-list(y0)[j]

    nu2=-math.log(1/nu-1); # y(wx+b)>=nu2

    w_dict={}
    for i in range(len(features)):
        w_dict[i]=w[0][i]

    bound=10

    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2a = {None : n_cat_bin},#n variables categoricas
            N2b= {None: n_cat_ord},
            #x0_0={None:0},
            ind ={None: n_ind},
            M3={None:100}, 
            k = {None: nu2},
            w=w_dict,
            b={None: b[0]},
            x0_1=x0_1,
            x0_2a=x0_2a,
            x0_2b=x0_2b,
            lowerOrd=lowerOrd,
            upperOrd=upperOrd,
            weight_discrete=weight_discrete,
            y=y_final,
            perc={None: perc},
            bound={None: bound},
            lam={None:lam},
            maxf ={None: maxf}
            )}


  
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")
    opt.options['TimeLimit'] = timelimit
    

      
    results = opt.solve(instance,tee=True) # tee=True: ver iteraciones por pantalla

    x_sol_aux=np.zeros([len(features),n_ind])

    for i in index_cont:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_1[i,j].value
    for i in index_cat_bin:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_2a[i,j].value

    for i in index_cat_ord:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_2b[i,j].value
    

    x_sol=pd.DataFrame(x_sol_aux,features)

    print(x_sol)


    cambio_x=np.zeros([len(features),n_ind])
    

    for i in range(len(features)):
        for j in range(n_ind):
            cambio_x[i,j]=x_sol_aux[i,j]-x0.iloc[j,i]
            if abs(cambio_x[i,j])<=1e-10:
                cambio_x[i,j]==0
    

    data=pd.DataFrame(cambio_x,index=features)

    data=data.transpose()


    return x_sol, data



def optimization_lineal_ind(i,x0,y0,lowerOrd,upperOrd,weight_discrete,model,w,b,index_cont,index_cat_bin,index_cat_ord,features_type,lam,nu):
   
  

    features=list(x0.columns)
    n_features=len(features)
    


    x0=x0.loc[i]
    y0=y0[i]

    x0_1={}
    x0_2a={}
    x0_2b={}

    cat_bin_features=[f for f in features if features_type[f]=='CatBinary']
    cat_ord_features=[f for f in features if features_type[f]=='CatOrdinal']
    cont_features=[f for f in features if features_type[f]=='Numerical']
    n_cat_bin=len(cat_bin_features)
    n_cat_ord=len(cat_ord_features)
    n_cont=len(cont_features)
   


    for (f,i) in zip(cont_features,index_cont):
         x0_1[i]=x0[f]
    for (f,i) in zip(cat_bin_features,index_cat_bin):
         x0_2a[i]=x0[f]
    for (f,i) in zip(cat_ord_features,index_cat_ord):
         x0_2b[i]=x0[f]


    nu2=-math.log(1/nu-1)

    w_dict={}
    for i in range(len(features)):
        w_dict[i]=w[0][i]


    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2a = {None : n_cat_bin},#n variables categoricas
            N2b= {None: n_cat_ord},
            M3={None:100}, 
            k = {None: nu2},
            w=w_dict,
            b={None: b[0]},
            x0_1=x0_1,
            x0_2a=x0_2a,
            x0_2b=x0_2b,
            lowerOrd=lowerOrd,
            upperOrd=upperOrd,
            weight_discrete=weight_discrete,
            y={None: -y0},
            lam={None:lam}
            )}


 
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")       
    results = opt.solve(instance,tee=True) 

    x_sol_aux=np.zeros(len(features))
    for i in index_cont:
        x_sol_aux[i]=instance.x_1[i].value
    for i in index_cat_bin:
        x_sol_aux[i]=instance.x_2a[i].value
    for i in index_cat_ord:
        x_sol_aux[i]=instance.x_2b[i].value
   


    x_sol=pd.DataFrame(x_sol_aux,features,columns=['x'])

    cambio_x=np.zeros(len(features))

    for j in range(len(features)):
        cambio_x[j]=x_sol_aux[j]-x0[j]



    return x_sol, cambio_x, -y0
