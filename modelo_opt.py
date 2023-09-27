# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:03:30 2020

@author: Jasone
"""


from __future__ import division
from pyomo.environ import *

def modelo_opt_rf_nonseparable(leaves,index_cont,index_cat_bin, index_cat_ord,var_related,objective):

    n_trees=len(leaves)
    #indiv=10 #n indv 
    model = AbstractModel()

    index_list = [] #index list (tree,leaf)
    for t in range(n_trees):
       for l in range(leaves[t]):
           index_list.append((t,l))

    if list(var_related.values())==[]:
        list_onehot=[]
        list_onehot2=[]
        n_onehot=0
    else:
        list_onehot2=[item for sublist in list(var_related.values()) for item in sublist]
        list_onehot=list(var_related.values())
        n_onehot=len(list_onehot)
        #index_onehot = [] #index list (tree,leaf)
        #for i in range(n_onehot):
        #   for l in list_onehot[i]:
        #       index_onehot.append((i,l))
    list_binary=list(set(index_cat_bin) - set(list_onehot2))
    
    index_nonehot=list(range(n_onehot))

    #parameters
    model.N1 = Param( within=Integers ) #continuos variables
    model.N2a = Param( within=Integers ) #categorical variables bin
    model.N2b = Param( within=Integers ) #categorical variables ord
    model.ind =Param (within=PositiveIntegers) #number of individuals
    model.Cont=Set(dimen=1,initialize=index_cont)
    model.CatBin=Set(dimen=1,initialize=index_cat_bin)
    model.CatOnehot=Set(dimen=1,initialize=list_onehot2)
    model.CatBinOnly=Set(dimen=1,initialize=list_binary)
    model.CatOrd=Set(dimen=1, initialize=index_cat_ord)
    model.CatOnehot_s=Set(dimen=1,initialize=index_nonehot)

    model.lowerOrd=Param(model.CatOrd,within=Integers)
    model.upperOrd=Param(model.CatOrd,within=Integers)
    model.weight_discrete=Param(model.CatOrd,within=Integers)

    model.I=RangeSet(0,model.ind-1)

    model.trees= Param (within =PositiveIntegers) #number of trees
    model.t=RangeSet(0,model.trees-1)
    model.leaves = Param(model.t) #number of leaves of each tree
    model.index_list = Set(dimen=2,initialize=index_list)
    model.values_leaf=Param(model.index_list)

    #parameters constraints
    model.nleft_num=Param(within=Integers) 
    model.nright_num=Param (within=Integers) 
    model.nleft_cat_bin=Param(within=Integers)
    model.nright_cat_bin=Param(within=Integers)
    model.nleft_cat_ord=Param(within=Integers)
    model.nright_cat_ord=Param(within=Integers)
    model.resleft_num=RangeSet(0,model.nleft_num-1) 
    model.resright_num= RangeSet(0,model.nright_num-1)
    model.resleft_cat_bin=RangeSet(0,model.nleft_cat_bin-1) 
    model.resright_cat_bin= RangeSet(0,model.nright_cat_bin-1)
    model.resleft_cat_ord=RangeSet(0,model.nleft_cat_ord-1) 
    model.resright_cat_ord= RangeSet(0,model.nright_cat_ord-1)
    model.left_num=Param(model.resleft_num,within=Any)
    model.right_num=Param(model.resright_num,within=Any)
    model.left_cat_bin=Param(model.resleft_cat_bin,within=Any)
    model.right_cat_bin=Param(model.resright_cat_bin,within=Any)
    model.left_cat_ord=Param(model.resleft_cat_ord,within=Any)
    model.right_cat_ord=Param(model.resright_cat_ord,within=Any)


    #bigMs
    model.M1=Param(within=PositiveReals) 
    model.M2=Param(within=Reals) 
    model.M3=Param(within=PositiveReals)

    model.epsi=Param(within=PositiveReals) #epsilon
    model.nu=Param(within=Reals)
    model.lam=Param(within=Reals)


    model.y_f= Param( model.I,within=Integers) # y=-y0 each ind
    model.x0_1=Param(model.Cont,model.I) #continuos x0
    model.x0_2a=Param(model.CatBin,model.I) #categorical binary x0
    model.x0_2b=Param(model.CatOrd,model.I)
    model.perc= Param(within=Integers) #number of individuals to be changed


    #variables
    #model.x_0 =Var(model.Fij, within=Reals) #bloque 0 que no se puede mover
    model.x_1 = Var( model.Cont, model.I ,bounds=(0,1)) 
    model.x_2a =Var (model.CatBin,model.I, within=Binary) #binary categ
    model.x_2b= Var(model.CatOrd,model.I, within=Integers) #binary ordinal


    model.z=Var(model.index_list,model.I,within=Binary)
    model.D=Var(model.t,model.I) 
    model.xi=Var(model.Cont,model.I,within=Binary) #aux l0 cont
    model.xi_c=Var(model.CatOrd,model.I,within=Binary) #aux l0 cat ord
    model.xi_b=Var(model.CatBinOnly,model.I,within=Binary) #aux l0 binarias 
    model.xi_o=Var(model.CatOnehot_s,model.I, within=Binary) #aux l0 onehot
    model.xi2=Var(model.Cont, within=Binary) #l0 global cont 
    model.xi2_c=Var(model.CatOrd, within=Binary)  #l0 global cat ord
    model.xi2_b=Var(model.CatBinOnly, within=Binary) #l0 global binarias
    model.xi2_o=Var(model.CatOnehot_s, within=Binary) #l0 global onehot
    model.w=Var(model.I,within=Binary) #w=1 if ind changes class
    model.phi=Var(model.I,within=Reals) #linearization auxiliar

  

    if objective=="l2l0ind":
        def obj_rule(model):
            return sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I)+model.lam*(sum(model.xi[n,i] for n in model.Cont for i in model.I)+sum(model.xi_c[n,i] for n in model.CatOrd for i in model.I)+sum(model.xi_b[n,i] for n in model.CatBinOnly for i in model.I)+sum(model.xi_o[i] for i in model.I))
        model.obj = Objective( rule=obj_rule )


    elif objective=="l2l0global":
        def obj_rule(model):
            return model.lam*(sum(model.xi2[n] for n in model.Cont)+sum(model.xi2_c[n] for n in model.CatOrd)+sum(model.xi2_b[n] for n in model.CatBinOnly)+sum(model.xi2_o[n] for n in model.CatOnehot_s))+sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I)+sum( ((model.x0_2a[n,i]-model.x_2a[n,i])/2)**2 for n in model.CatBin for i in model.I)+sum( ((model.x0_2b[n,i]-model.x_2b[n,i])/model.weight_discrete[n])**2 for n in model.CatOrd for i in model.I)
        model.obj = Objective( rule=obj_rule )



    #RangeSet is 1-based RangeSet(5)=[1,2,3,4,5]

    #constraints


    def path_left_num(model,s,i):
        return model.x_1[model.left_num[s][2],i]-(model.M1-model.left_num[s][3])*(1-model.z[model.left_num[s][0],model.left_num[s][1],i])+model.epsi<=model.left_num[s][3]
    model.pathleft_num= Constraint(model.resleft_num,model.I,rule=path_left_num)

    def path_right_num(model,s,i):
        return model.x_1[model.right_num[s][2],i]+(model.M2+model.right_num[s][3])*(1-model.z[model.right_num[s][0],model.right_num[s][1],i])-model.epsi>=model.right_num[s][3]
    model.pathright_num= Constraint(model.resright_num,model.I,rule=path_right_num)


    def path_left_cat_bin(model,s,i):
        return model.x_2a[model.left_cat_bin[s][2],i]-(model.M1-model.left_cat_bin[s][3])*(1-model.z[model.left_cat_bin[s][0],model.left_cat_bin[s][1],i])+model.epsi<=model.left_cat_bin[s][3]
    model.pathleft_cat_bin= Constraint(model.resleft_cat_bin,model.I,rule=path_left_cat_bin)

    def path_right_cat_bin(model,s,i):
        return model.x_2a[model.right_cat_bin[s][2],i]+(model.M2+model.right_cat_bin[s][3])*(1-model.z[model.right_cat_bin[s][0],model.right_cat_bin[s][1],i])-model.epsi>=model.right_cat_bin[s][3]
    model.pathright_cat_bin= Constraint(model.resright_cat_bin,model.I,rule=path_right_cat_bin)

    def path_left_cat_ord(model,s,i):
        return model.x_2b[model.left_cat_ord[s][2],i]-(model.M1-model.left_cat_ord[s][3])*(1-model.z[model.left_cat_ord[s][0],model.left_cat_ord[s][1],i])+model.epsi<=model.left_cat_ord[s][3]
    model.pathleft_cat_ord= Constraint(model.resleft_cat_ord,model.I,rule=path_left_cat_ord)

    def path_right_cat_ord(model,s,i):
        return model.x_2b[model.right_cat_ord[s][2],i]+(model.M2+model.right_cat_ord[s][3])*(1-model.z[model.right_cat_ord[s][0],model.right_cat_ord[s][1],i])-model.epsi>=model.right_cat_ord[s][3]
    model.pathright_cat_ord= Constraint(model.resright_cat_ord,model.I,rule=path_right_cat_ord)


    def one_path(model,t,i):
        return sum(model.z[t,l,i] for l in RangeSet(0,model.leaves[t]-1))==1.0
    model.path=Constraint(model.t,model.I,rule=one_path)

    def def_output(model,t,i):
        return model.D[t,i]==sum(model.values_leaf[t,l]*model.z[t,l,i] for l in RangeSet(0,model.leaves[t]-1))
    model.output=Constraint(model.t,model.I,rule=def_output)

    def def_finalclass(model,i):
        return model.y_f[i]*model.phi[i]>=model.nu
    model.finalclass=Constraint(model.I, rule=def_finalclass) 

    def aux_group1(model,i):
        return -model.w[i]*model.trees<=model.phi[i]
    model.auxg1=Constraint(model.I, rule=aux_group1)

    def aux_group2(model,i):
        return model.phi[i]<=model.w[i]*model.trees
    model.auxg2=Constraint(model.I, rule=aux_group2)

    def aux_group3(model,i):
        return sum(model.D[t,i] for t in model.t)-(1-model.w[i])*model.trees<=model.phi[i]
    model.auxg3=Constraint(model.I, rule=aux_group3)

    def aux_group4(model,i):
        return model.phi[i]<=sum(model.D[t,i] for t in model.t)+(1-model.w[i])*model.trees
    model.auxg4=Constraint(model.I, rule=aux_group4)

    def ind_change(model):
        return sum(model.w[i] for i in model.I)>=model.perc #number of individuals to be changed
    model.indcambio=Constraint(rule=ind_change)


     #l0 ind continua
    def aux_l01(model,n,i):
        return -model.M3*model.xi[n,i]<=(model.x_1[n,i]-model.x0_1[n,i])
    model.auxl01=Constraint(model.Cont,model.I,rule=aux_l01)

    def aux_l02(model,n,i):
        return (model.x_1[n,i]-model.x0_1[n,i])<=model.xi[n,i]*model.M3
    model.auxl02=Constraint(model.Cont,model.I,rule=aux_l02)

    #l0 ind cat ordinal
    def aux_l03(model,n,i):
        return -model.M3*model.xi_c[n,i]<=(model.x_2b[n,i]-model.x0_2b[n,i])
    model.auxl03=Constraint(model.CatOrd,model.I,rule=aux_l03)

    def aux_l04(model,n,i):
        return (model.x_2b[n,i]-model.x0_2b[n,i])<=model.xi_c[n,i]*model.M3
    model.auxl04=Constraint(model.CatOrd,model.I,rule=aux_l04)

    #l0 ind cat bin
    def aux_l05(model,n,i):
        return -model.M3*model.xi_b[n,i]<=(model.x_2a[n,i]-model.x0_2a[n,i])
    model.auxl05=Constraint(model.CatBinOnly,model.I,rule=aux_l05)

    def aux_l06(model,n,i):
        return (model.x_2a[n,i]-model.x0_2a[n,i])<=model.xi_b[n,i]*model.M3
    model.auxl06=Constraint(model.CatBinOnly,model.I,rule=aux_l06)

    #l0 ind cat one-hot (esta hecho para cuando tengo solo 1 variable one-hot)
    def aux_l07(model,n,s,i):
        if s in list_onehot[n]:
            return model.xi_o[n,i]>=(model.x_2a[s,i]-model.x0_2a[s,i])**2
        else:
            return model.xi_o[n,i]<=1
    model.auxl07=Constraint(model.CatOnehot_s,model.CatOnehot,model.I,rule=aux_l07)


    #l0 globales

    def aux_l0global1(model,n,i):
        return model.xi2[n] >= model.xi[n,i]
    model.auxl0g1=Constraint(model.Cont,model.I,rule=aux_l0global1)

    def aux_l0globalc(model,n,i):
        return model.xi2_c[n] >= model.xi_c[n,i]
    model.auxl0g2=Constraint(model.CatOrd,model.I,rule=aux_l0globalc)

    def aux_l0globalb(model,n,i):
        return model.xi2_b[n] >= model.xi_b[n,i]
    model.auxl0g3=Constraint(model.CatBinOnly,model.I,rule=aux_l0globalb)


    def aux_l0globalo(model,n,i):
        return model.xi2_o[n]>= model.xi_o[n,i]
    model.auxl0g4=Constraint(model.CatOnehot_s,model.I,rule=aux_l0globalo)


    if list_onehot!=[]:
        def aux_onehot(model,s,i):
            return sum(model.x_2a[n,i] for n in list_onehot[s])==1
        model.auxonehot=Constraint(model.CatOnehot_s,model.I,rule=aux_onehot)

    #def aux_l11(model,n):#
    #       return model.xi[n]>=(model.x0_1[n]-model.x_1[n])
    #model.auxl11=Constraint(model.Cont,rule=aux_l11)

    #def aux_l12(model,n):
    #    return -model.xi[n]<=(model.x0_1[n]-model.x_1[n])
    #model.auxl12=Constraint(model.Cont,rule=aux_l12)

    #bounds cat ord
    def bounds_ord(model,n,i):
        return model.x_2b[n,i] <= model.upperOrd[n]
    model.bounds=Constraint(model.CatOrd,model.I,rule=bounds_ord)

    def bounds_ord2(model,n,i):
        return model.x_2b[n,i] >= model.lowerOrd[n]
    model.bounds2=Constraint(model.CatOrd,model.I,rule=bounds_ord2)

    return model


def modelo_opt_rf_separable(leaves,index_cont,index_cat_bin,index_cat_ord,var_related,objective):

  

    n_trees=len(leaves)
    #indiv=10 #n indv 
    model = AbstractModel()

    index_list = [] #index list (tree,leaf)
    for t in range(n_trees):
       for l in range(leaves[t]):
           index_list.append((t,l))

    if list(var_related.values())==[]:
        list_onehot=[]
        list_onehot2=[]
        n_onehot=0
    else:
        list_onehot2=[item for sublist in list(var_related.values()) for item in sublist]
        list_onehot=list(var_related.values())
        n_onehot=len(list_onehot)
        #index_onehot = [] #index list (tree,leaf)
        #for i in range(n_onehot):
        #   for l in list_onehot[i]:
        #       index_onehot.append((i,l))
    list_binary=list(set(index_cat_bin) - set(list_onehot2))
    
    index_nonehot=list(range(n_onehot))

    #parameters
    
    model.N1 = Param( within=Integers ) #continuos variables
    model.N2a = Param( within=Integers ) #categorical variables bin
    model.N2b = Param( within=Integers ) #categorical variables ord
    model.Cont=Set(dimen=1,initialize=index_cont)
    model.CatBin=Set(dimen=1,initialize=index_cat_bin)
    model.CatOnehot=Set(dimen=1,initialize=list_onehot2)
    model.CatBinOnly=Set(dimen=1,initialize=list_binary)
    model.CatOrd=Set(dimen=1, initialize=index_cat_ord)
    model.CatOnehot_s=Set(dimen=1,initialize=index_nonehot)
   
    model.lowerOrd=Param(model.CatOrd,within=Integers)
    model.upperOrd=Param(model.CatOrd,within=Integers)
    model.weight_discrete=Param(model.CatOrd,within=Integers)
    
    model.trees= Param (within =PositiveIntegers) #number of trees
    model.t=RangeSet(0,model.trees-1)
    model.leaves = Param(model.t) #number of leaves of each tree
    model.index_list = Set(dimen=2,initialize=index_list)
    model.values_leaf=Param(model.index_list)

    #parameters constraints
    model.nleft_num=Param(within=Integers) 
    model.nright_num=Param (within=Integers) 
    model.nleft_cat_bin=Param(within=Integers)
    model.nright_cat_bin=Param(within=Integers)
    model.nleft_cat_ord=Param(within=Integers)
    model.nright_cat_ord=Param(within=Integers)
    model.resleft_num=RangeSet(0,model.nleft_num-1) 
    model.resright_num= RangeSet(0,model.nright_num-1)
    model.resleft_cat_bin=RangeSet(0,model.nleft_cat_bin-1) 
    model.resright_cat_bin= RangeSet(0,model.nright_cat_bin-1)
    model.resleft_cat_ord=RangeSet(0,model.nleft_cat_ord-1) 
    model.resright_cat_ord= RangeSet(0,model.nright_cat_ord-1)
    model.left_num=Param(model.resleft_num,within=Any)
    model.right_num=Param(model.resright_num,within=Any)
    model.left_cat_bin=Param(model.resleft_cat_bin,within=Any)
    model.right_cat_bin=Param(model.resright_cat_bin,within=Any)
    model.left_cat_ord=Param(model.resleft_cat_ord,within=Any)
    model.right_cat_ord=Param(model.resright_cat_ord,within=Any)

    #bigMs
    model.M1=Param(within=PositiveReals) 
    model.M2=Param(within=Reals) 
    model.M3=Param(within=PositiveReals)

    model.epsi=Param(within=PositiveReals) #epsilon
    model.nu=Param(within=Reals)
    model.lam=Param(within=PositiveReals) #lambda from lambda*l0+l2

    model.y= Param( within=Integers) # y=-y0 each ind
    model.x0_1=Param(model.Cont) #continuos x0
    model.x0_2a=Param(model.CatBin) #categorical binary x0
    model.x0_2b=Param(model.CatOrd)


    #variables
    #model.x_0 =Var(model.Fij, within=Reals) #bloque 0 que no se puede mover
    model.x_1 = Var( model.Cont,bounds=(0,1) ) 
    model.x_2a =Var (model.CatBin, within=Binary) #binary categ
    model.x_2b= Var(model.CatOrd,within=Integers) #binary ordinal


    model.z=Var(model.index_list,within=Binary)
    model.D=Var(model.t) 
    model.xi=Var(model.Cont,within=Binary) #aux l0 cont
    model.xi_c=Var(model.CatOrd,within=Binary) #aux l0 cat ord
    model.xi_b=Var(model.CatBinOnly,within=Binary) #aux l0 binarias 
    model.xi_o=Var(model.CatOnehot_s,within=Binary) #aux l0 onehot

    
    if objective=='l2':
        def obj_rule(model):
            return sum( (model.x0_1[n]-model.x_1[n])**2 for n in model.Cont)+sum((model.x0_2a[n]-model.x_2a[n])**2 for n in model.CatBin)+sum((model.x0_2b[n]-model.x_2b[n])**2 for n in model.CatOrd)
        model.obj = Objective( rule=obj_rule )

    elif objective=='l0':
        def obj_rule(model):
            return sum(model.xi[n] for n in model.Cont)+sum(model.xi_c[n] for n in model.CatOrd)+sum(model.xi_b[n] for n in model.CatBinOnly)+model.xi_o
        model.obj = Objective( rule=obj_rule )

    elif objective=='l0l2':
        def obj_rule(model):
             return model.lam*(sum(model.xi[n] for n in model.Cont)+sum(model.xi_c[n] for n in model.CatOrd)+sum(model.xi_b[n] for n in model.CatBinOnly)+sum(model.xi_o[n] for n in model.CatOnehot_s))+(sum( (model.x0_1[n]-model.x_1[n])**2 for n in model.Cont))+(sum( ((model.x0_2a[n]-model.x_2a[n])/2)**2 for n in model.CatBin))+(sum( ((model.x0_2b[n]-model.x_2b[n])/model.weight_discrete[n])**2 for n in model.CatOrd))
        model.obj = Objective( rule=obj_rule )




    #RangeSet is 1-based RangeSet(5)=[1,2,3,4,5]

    #constraints

    def path_left_num(model,s):
        return model.x_1[model.left_num[s][2]]-(model.M1-model.left_num[s][3])*(1-model.z[model.left_num[s][0],model.left_num[s][1]])+model.epsi<=model.left_num[s][3]
    model.pathleft_num= Constraint(model.resleft_num,rule=path_left_num)

    def path_right_num(model,s):
        return model.x_1[model.right_num[s][2]]+(model.M2+model.right_num[s][3])*(1-model.z[model.right_num[s][0],model.right_num[s][1]])-model.epsi>=model.right_num[s][3]
    model.pathright_num= Constraint(model.resright_num,rule=path_right_num)


    def path_left_cat_bin(model,s):
        return model.x_2a[model.left_cat_bin[s][2]]-(model.M1-model.left_cat_bin[s][3])*(1-model.z[model.left_cat_bin[s][0],model.left_cat_bin[s][1]])+model.epsi<=model.left_cat_bin[s][3]
    model.pathleft_cat_bin= Constraint(model.resleft_cat_bin,rule=path_left_cat_bin)

    def path_right_cat_bin(model,s):
        return model.x_2a[model.right_cat_bin[s][2]]+(model.M2+model.right_cat_bin[s][3])*(1-model.z[model.right_cat_bin[s][0],model.right_cat_bin[s][1]])-model.epsi>=model.right_cat_bin[s][3]
    model.pathright_cat_bin= Constraint(model.resright_cat_bin,rule=path_right_cat_bin)

    def path_left_cat_ord(model,s):
        return model.x_2b[model.left_cat_ord[s][2]]-(model.M1-model.left_cat_ord[s][3])*(1-model.z[model.left_cat_ord[s][0],model.left_cat_ord[s][1]])+model.epsi<=model.left_cat_ord[s][3]
    model.pathleft_cat_ord= Constraint(model.resleft_cat_ord,rule=path_left_cat_ord)

    def path_right_cat_ord(model,s):
        return model.x_2b[model.right_cat_ord[s][2]]+(model.M2+model.right_cat_ord[s][3])*(1-model.z[model.right_cat_ord[s][0],model.right_cat_ord[s][1]])-model.epsi>=model.right_cat_ord[s][3]
    model.pathright_cat_ord= Constraint(model.resright_cat_ord,rule=path_right_cat_ord)

    def one_path(model,t):
        return sum(model.z[t,l] for l in RangeSet(0,model.leaves[t]-1))==1.0
    model.path=Constraint(model.t,rule=one_path)

    def def_salida(model,t):
        return model.D[t]==sum(model.values_leaf[t,l]*model.z[t,l] for l in RangeSet(0,model.leaves[t]-1))
    model.salida=Constraint(model.t,rule=def_salida)

    def def_clase(model):
        return model.y*(sum(model.D[t] for t in model.t))>=model.nu
    model.clase=Constraint(rule=def_clase)

    def aux_l01(model,n):
        return -model.M3*model.xi[n]<=(model.x_1[n]-model.x0_1[n])
    model.auxl01=Constraint(model.Cont,rule=aux_l01)

    def aux_l02(model,n):
        return (model.x_1[n]-model.x0_1[n])<=model.xi[n]*model.M3
    model.auxl02=Constraint(model.Cont,rule=aux_l02)

    #l0 ind cat ordinal
    def aux_l03(model,n):
        return -model.M3*model.xi_c[n]<=(model.x_2b[n]-model.x0_2b[n])
    model.auxl03=Constraint(model.CatOrd,rule=aux_l03)

    def aux_l04(model,n):
        return (model.x_2b[n]-model.x0_2b[n])<=model.xi_c[n]*model.M3
    model.auxl04=Constraint(model.CatOrd,rule=aux_l04)

    #l0 ind cat bin
    def aux_l05(model,n):
        return -model.M3*model.xi_b[n]<=(model.x_2a[n]-model.x0_2a[n])
    model.auxl05=Constraint(model.CatBinOnly,rule=aux_l05)

    def aux_l06(model,n):
        return (model.x_2a[n]-model.x0_2a[n])<=model.xi_b[n]*model.M3
    model.auxl06=Constraint(model.CatBinOnly,rule=aux_l06)

    #l0 ind cat one-hot 
    def aux_l07(model,n,s):
        if s in list_onehot[n]:
            return model.xi_o[n]>=(model.x_2a[s]-model.x0_2a[s])**2
        else:
            return model.xi_o[n]<=1
    model.auxl07=Constraint(model.CatOnehot_s,model.CatOnehot,rule=aux_l07)



    if list_onehot!=[]:
        def aux_onehot(model,s):
            return sum(model.x_2a[n] for n in list_onehot[s])==1
        model.auxonehot=Constraint(model.CatOnehot_s,rule=aux_onehot)

    #bounds cat ord
    def bounds_ord(model,n):
        return model.x_2b[n] <= model.upperOrd[n]
    model.bounds=Constraint(model.CatOrd,rule=bounds_ord)

    def bounds_ord2(model,n):
        return model.x_2b[n] >= model.lowerOrd[n]
    model.bounds2=Constraint(model.CatOrd,rule=bounds_ord2)
   


    #def aux_l11(model,n):#
    #       return model.xi[n]>=(model.x0_1[n]-model.x_1[n])
    #model.auxl11=Constraint(model.Cont,rule=aux_l11)

    #def aux_l12(model,n):
    #    return -model.xi[n]<=(model.x0_1[n]-model.x_1[n])
    #model.auxl12=Constraint(model.Cont,rule=aux_l12)


    return model


def modelo_opt_lineal_nonseparable(index_cont,index_cat_bin, index_cat_ord,var_related,objective,l0globcons):

    
    model = AbstractModel()


    if list(var_related.values())==[]:
        list_onehot=[]
        list_onehot2=[]
        n_onehot=0
    else:
        list_onehot2=[item for sublist in list(var_related.values()) for item in sublist]
        list_onehot=list(var_related.values())
        n_onehot=len(list_onehot)
        #index_onehot = [] #index list (tree,leaf)
        #for i in range(n_onehot):
        #   for l in list_onehot[i]:
        #       index_onehot.append((i,l))
    list_binary=list(set(index_cat_bin) - set(list_onehot2))
    
    index_nonehot=list(range(n_onehot))


    #parameters
    #model.N0 = Param( within=Integers ) #unmovable
    model.N1 = Param( within=Integers ) #continuos variables
    model.N2a = Param( within=Integers ) #categorical variables bin
    model.N2b = Param( within=Integers ) #categorical variables ord
    model.ind =Param (within=Integers) #number of individuals
    #model.Fij = RangeSet (model.N0) 
    model.Cont=Set(dimen=1,initialize=index_cont)
    model.CatBin=Set(dimen=1,initialize=index_cat_bin)
    model.CatOnehot=Set(dimen=1,initialize=list_onehot2)
    model.CatBinOnly=Set(dimen=1,initialize=list_binary)
    model.CatOrd=Set(dimen=1, initialize=index_cat_ord)
    model.CatOnehot_s=Set(dimen=1,initialize=index_nonehot)
    model.I=RangeSet(0,model.ind-1)

    model.lowerOrd=Param(model.CatOrd,within=Integers)
    model.upperOrd=Param(model.CatOrd,within=Integers)
    model.weight_discrete=Param(model.CatOrd,within=Integers)

    #model parameters
    model.w = Param( RangeSet(0,model.N1+model.N2a+model.N2b) ) #weights
    model.b = Param( within=Reals ) #bias
    model.k = Param( within=Reals) #threshold
    model.maxf = Param (within= PositiveReals) 

    


    #bigMs
   
    model.M3=Param(within=PositiveReals)
    model.lam=Param(within=Reals)
    model.bound=Param(within=Reals)

    model.y= Param( model.I,within=Integers) # y=-y0 each ind
    #model.x0_0 =Param(model.Fij) #unmovable x0
    model.x0_1=Param(model.Cont,model.I) #continuos x0
    model.x0_2a=Param(model.CatBin,model.I) #categorical binary x0
    model.x0_2b=Param(model.CatOrd,model.I)
    model.perc= Param(within=Integers) #number of individuals to be changed


    #variables
    #model.x_0 =Var(model.Fij, within=Reals) #bloque 0 que no se puede mover
    model.x_1 = Var( model.Cont, model.I,bounds=(0,1)) 
    model.x_2a =Var (model.CatBin,model.I, within=Binary) #binary categ
    model.x_2b= Var(model.CatOrd,model.I, within=Integers) #binary ordinal


  
    model.xi=Var(model.Cont,model.I,within=Binary) #aux l0 cont
    model.xi_c=Var(model.CatOrd,model.I,within=Binary) #aux l0 cat ord
    model.xi_b=Var(model.CatBinOnly,model.I,within=Binary) #aux l0 binarias 
    model.xi_o=Var(model.CatOnehot_s,model.I, within=Binary) #aux l0 onehot
    model.xi2=Var(model.Cont, within=Binary) #l0 global cont 
    model.xi2_c=Var(model.CatOrd, within=Binary)  #l0 global cat ord
    model.xi2_b=Var(model.CatBinOnly, within=Binary) #l0 global binarias
    model.xi2_o=Var(model.CatOnehot_s, within=Binary) #l0 global onehot

    model.q=Var(model.I,within=Binary) #w=1 if ind changes class
    model.phi=Var(model.I,within=Reals) #linearization auxiliar

  

   
    if objective=="l2l0ind":
        def obj_rule(model):
            return sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I)+model.lam*(sum(model.xi[n,i] for n in model.Cont for i in model.I)+sum(model.xi_c[n,i] for n in model.CatOrd for i in model.I)+sum(model.xi_b[n,i] for n in model.CatBinOnly for i in model.I)+sum(model.xi_o[i] for i in model.I))
        model.obj = Objective( rule=obj_rule )


    elif objective=="l2l0global":
        def obj_rule(model):
            return model.lam*(sum(model.xi2[n] for n in model.Cont)+sum(model.xi2_c[n] for n in model.CatOrd)+sum(model.xi2_b[n] for n in model.CatBinOnly)+sum(model.xi2_o[n] for n in model.CatOnehot_s))+sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I)+sum( ((model.x0_2a[n,i]-model.x_2a[n,i])/2)**2 for n in model.CatBin for i in model.I)+sum( ((model.x0_2b[n,i]-model.x_2b[n,i])/model.weight_discrete[n])**2 for n in model.CatOrd for i in model.I)
        model.obj = Objective( rule=obj_rule )

    elif objective=="l2":
        def obj_rule(model):
            return (sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I))+(sum( ((model.x0_2a[n,i]-model.x_2a[n,i])/2)**2 for n in model.CatBin for i in model.I))+(sum( ((model.x0_2b[n,i]-model.x_2b[n,i])/model.weight_discrete[n])**2 for n in model.CatOrd for i in model.I))
        model.obj = Objective( rule=obj_rule )


 

    #RangeSet is 1-based RangeSet(5)=[1,2,3,4,5]

    #constraints

    if l0globcons=="True":

        def l0glob(model): 
            return sum(model.xi2[n] for n in model.Cont)+sum(model.xi2_c[n] for n in model.CatOrd)+sum(model.xi2_b[n] for n in model.CatBinOnly)+sum(model.xi2_o[n] for n in model.CatOnehot_s)<= model.maxf
        model.maxl0g = Constraint (rule= l0glob)

    
    def clase_rule(model,i):
        return  model.y[i]*model.phi[i]>=model.k*model.q[i]
    model.clase = Constraint (model.I,rule=clase_rule)

  


    def aux_group1(model,i):
        return -model.q[i]*model.bound<=model.phi[i]
    model.auxg1=Constraint(model.I, rule=aux_group1)

    def aux_group2(model,i):
        return model.phi[i]<=model.q[i]*model.bound
    model.auxg2=Constraint(model.I, rule=aux_group2)


    def aux_group3(model,i):
        return (sum(model.w[n]*model.x_1[n,i] for n in model.Cont)+sum(model.w[s]*model.x_2a[s,i] for s in model.CatBin)+sum(model.w[s]*model.x_2b[s,i] for s in model.CatOrd)+model.b)-(1-model.q[i])*model.bound<=model.phi[i]
    model.auxg3=Constraint(model.I, rule=aux_group3)

    def aux_group4(model,i):
        return model.phi[i]<=(sum(model.w[n]*model.x_1[n,i] for n in model.Cont)+sum(model.w[s]*model.x_2a[s,i] for s in model.CatBin)+sum(model.w[s]*model.x_2b[s,i] for s in model.CatOrd)+model.b)+(1-model.q[i])*model.bound
    model.auxg4=Constraint(model.I, rule=aux_group4)

    def ind_change(model):
        return sum(model.q[i] for i in model.I)>=model.perc #number of individuals to be changed
    model.indcambio=Constraint(rule=ind_change)

    #l0 ind continua
    def aux_l01(model,n,i):
        return -model.M3*model.xi[n,i]<=(model.x_1[n,i]-model.x0_1[n,i])
    model.auxl01=Constraint(model.Cont,model.I,rule=aux_l01)

    def aux_l02(model,n,i):
        return (model.x_1[n,i]-model.x0_1[n,i])<=model.xi[n,i]*model.M3
    model.auxl02=Constraint(model.Cont,model.I,rule=aux_l02)

    #l0 ind cat ordinal
    def aux_l03(model,n,i):
        return -model.M3*model.xi_c[n,i]<=(model.x_2b[n,i]-model.x0_2b[n,i])
    model.auxl03=Constraint(model.CatOrd,model.I,rule=aux_l03)

    def aux_l04(model,n,i):
        return (model.x_2b[n,i]-model.x0_2b[n,i])<=model.xi_c[n,i]*model.M3
    model.auxl04=Constraint(model.CatOrd,model.I,rule=aux_l04)

    #l0 ind cat bin
    def aux_l05(model,n,i):
        return -model.M3*model.xi_b[n,i]<=(model.x_2a[n,i]-model.x0_2a[n,i])
    model.auxl05=Constraint(model.CatBinOnly,model.I,rule=aux_l05)

    def aux_l06(model,n,i):
        return (model.x_2a[n,i]-model.x0_2a[n,i])<=model.xi_b[n,i]*model.M3
    model.auxl06=Constraint(model.CatBinOnly,model.I,rule=aux_l06)

    #l0 ind cat one-hot 
   
    def aux_l07(model,n,s,i):
        if s in list_onehot[n]:
            return model.xi_o[n,i]>=(model.x_2a[s,i]-model.x0_2a[s,i])**2
        else:
            return model.xi_o[n,i]<=1
    model.auxl07=Constraint(model.CatOnehot_s,model.CatOnehot,model.I,rule=aux_l07)



    #l0 globales

    def aux_l0global1(model,n,i):
        return model.xi2[n] >= model.xi[n,i]
    model.auxl0g1=Constraint(model.Cont,model.I,rule=aux_l0global1)

    def aux_l0globalc(model,n,i):
        return model.xi2_c[n] >= model.xi_c[n,i]
    model.auxl0g2=Constraint(model.CatOrd,model.I,rule=aux_l0globalc)

    def aux_l0globalb(model,n,i):
        return model.xi2_b[n] >= model.xi_b[n,i]
    model.auxl0g3=Constraint(model.CatBinOnly,model.I,rule=aux_l0globalb)


    def aux_l0globalo(model,n,i):
        return model.xi2_o[n]>= model.xi_o[n,i]
    model.auxl0g4=Constraint(model.CatOnehot_s,model.I,rule=aux_l0globalo)

    if list_onehot!=[]:
        def aux_onehot(model,s,i):
            return sum(model.x_2a[n,i] for n in list_onehot[s])==1
        model.auxonehot=Constraint(model.CatOnehot_s,model.I,rule=aux_onehot)


    #bounds cat ord
    def bounds_ord(model,n,i):
        return model.x_2b[n,i] <= model.upperOrd[n]
    model.bounds=Constraint(model.CatOrd,model.I,rule=bounds_ord)

    def bounds_ord2(model,n,i):
        return model.x_2b[n,i] >= model.lowerOrd[n]
    model.bounds2=Constraint(model.CatOrd,model.I,rule=bounds_ord2)

    return model


def modelo_opt_lineal_separable(index_cont,index_cat_bin, index_cat_ord,var_related,objective):

    model = AbstractModel()

    if list(var_related.values())==[]:
        list_onehot=[]
        list_onehot2=[]
        n_onehot=0
    else:
        list_onehot2=[item for sublist in list(var_related.values()) for item in sublist]
        list_onehot=list(var_related.values())
        n_onehot=len(list_onehot)
        #index_onehot = [] #index list (tree,leaf)
        #for i in range(n_onehot):
        #   for l in list_onehot[i]:
        #       index_onehot.append((i,l))
    list_binary=list(set(index_cat_bin) - set(list_onehot2))
    
    index_nonehot=list(range(n_onehot))


    #parametros
    model.N1 = Param( within=Integers ) #continuos
    model.N2a = Param( within=Integers ) #categorical bin
    model.N2b = Param( within=Integers ) #categorical ord

   
    model.Cont=Set(dimen=1,initialize=index_cont)
    model.CatBin=Set(dimen=1,initialize=index_cat_bin)
    model.CatOnehot=Set(dimen=1,initialize=list_onehot2)
    model.CatBinOnly=Set(dimen=1,initialize=list_binary)
    model.CatOrd=Set(dimen=1, initialize=index_cat_ord)
    model.CatOnehot_s=Set(dimen=1,initialize=index_nonehot)

    #model.index_onehot = Set(dimen=2,initialize=index_onehot)

    model.lowerOrd=Param(model.CatOrd,within=Integers)
    model.upperOrd=Param(model.CatOrd,within=Integers)
    model.weight_discrete=Param(model.CatOrd,within=Integers)

    model.w = Param( RangeSet(0,model.N1+model.N2a+model.N2b) ) #weights
    model.b = Param( within=Reals ) #bias
    model.k = Param( within=Reals) #threshold
    

    model.y= Param( within=Integers)
    model.x0_1=Param(model.Cont) #continuos x0
    model.x0_2a=Param(model.CatBin) #categorical binary x0
    model.x0_2b=Param(model.CatOrd)


    model.lam=Param(within=PositiveReals) #lambda from lambda*l0+l2
    model.M3=Param(within=PositiveReals)

    #variables 
    model.x_1 = Var( model.Cont, within=Reals,bounds=(0,1) ) 
    model.x_2a =Var (model.CatBin, within=Binary) #binary categ
    model.x_2b= Var(model.CatOrd, within=Integers) #binary ordinal
    model.xi =Var(model.Cont,within=Binary)
    model.xi_c=Var(model.CatOrd,within=Binary) #aux l0 cat ord
    model.xi_b=Var(model.CatBinOnly,within=Binary) #aux l0 binarias 
    model.xi_o=Var(model.CatOnehot_s, within=Binary) #aux l0 onehot


    if objective=='l2':

        def obj_rule(model):
            return (sum( (model.x0_1[n]-model.x_1[n])**2 for n in model.Cont))+(sum( ((model.x0_2a[n]-model.x_2a[n])/2)**2 for n in model.CatBin))+(sum( ((model.x0_2b[n]-model.x_2b[n])/model.weight_discrete[n])**2 for n in model.CatOrd))
        model.obj = Objective( rule=obj_rule )

    elif objective=='l0':
        def obj_rule(model):
            return sum(model.xi[n] for n in model.Cont)+sum(model.xi_c[n] for n in model.CatOrd)+sum(model.xi_b[n] for n in model.CatBinOnly)+sum(model.xi_o[n] for n in model.CatOnehot_s)
        model.obj = Objective( rule=obj_rule )

    elif objective=='l0l2':
        def obj_rule(model):
             return model.lam*(sum(model.xi[n] for n in model.Cont)+sum(model.xi_c[n] for n in model.CatOrd)+sum(model.xi_b[n] for n in model.CatBinOnly)+sum(model.xi_o[n] for n in model.CatOnehot_s))+(sum( (model.x0_1[n]-model.x_1[n])**2 for n in model.Cont))+(sum( ((model.x0_2a[n]-model.x_2a[n])/2)**2 for n in model.CatBin))+(sum( ((model.x0_2b[n]-model.x_2b[n])/model.weight_discrete[n])**2 for n in model.CatOrd))
        model.obj = Objective( rule=obj_rule )



    def clase_rule(model):
        return  model.y*(sum(model.w[n]*model.x_1[n] for n in model.Cont)+sum(model.w[s]*model.x_2a[s] for s in model.CatBin)+sum(model.w[s]*model.x_2b[s] for s in model.CatOrd)+model.b)>=model.k
    model.clase = Constraint (rule=clase_rule)


    def aux_l01(model,n):
        return -model.M3*model.xi[n]<=(model.x_1[n]-model.x0_1[n])
    model.auxl01=Constraint(model.Cont,rule=aux_l01)

    def aux_l02(model,n):
        return (model.x_1[n]-model.x0_1[n])<=model.xi[n]*model.M3
    model.auxl02=Constraint(model.Cont,rule=aux_l02)

    #l0 ind cat ordinal
    def aux_l03(model,n):
        return -model.M3*model.xi_c[n]<=(model.x_2b[n]-model.x0_2b[n])
    model.auxl03=Constraint(model.CatOrd,rule=aux_l03)

    def aux_l04(model,n):
        return (model.x_2b[n]-model.x0_2b[n])<=model.xi_c[n]*model.M3
    model.auxl04=Constraint(model.CatOrd,rule=aux_l04)

    #l0 ind cat bin
    def aux_l05(model,n):
        return -model.M3*model.xi_b[n]<=(model.x_2a[n]-model.x0_2a[n])
    model.auxl05=Constraint(model.CatBinOnly,rule=aux_l05)

    def aux_l06(model,n):
        return (model.x_2a[n]-model.x0_2a[n])<=model.xi_b[n]*model.M3
    model.auxl06=Constraint(model.CatBinOnly,rule=aux_l06)


    
    #l0 ind cat one-hot 
    def aux_l07(model,n,s):
        if s in list_onehot[n]:
            return model.xi_o[n]>=(model.x_2a[s]-model.x0_2a[s])**2
        else:
            return model.xi_o[n]<=1
    model.auxl07=Constraint(model.CatOnehot_s,model.CatOnehot,rule=aux_l07)



    if list_onehot!=[]:
        def aux_onehot(model,s):
            return sum(model.x_2a[n] for n in list_onehot[s])==1
        model.auxonehot=Constraint(model.CatOnehot_s,rule=aux_onehot)

    #bounds cat ord
    def bounds_ord(model,n):
        return model.x_2b[n] <= model.upperOrd[n]
    model.bounds=Constraint(model.CatOrd,rule=bounds_ord)

    def bounds_ord2(model,n):
        return model.x_2b[n] >= model.lowerOrd[n]
    model.bounds2=Constraint(model.CatOrd,rule=bounds_ord2)

    return model



