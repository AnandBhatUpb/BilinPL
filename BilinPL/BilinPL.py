# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:29:20 2018

@author: Anand
"""
 
from py4j.java_gateway import JavaGateway
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import scipy.stats as stats
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

JavaGateway.launch_gateway
gateway = JavaGateway()                          # connect to the JVM
java_object = gateway.entry_point

def convertIntToDouble(rank):                    # converts integer to double
    len_i = len(rank)
    len_j = len(rank[0])
    for i in range(len_i):
        for j in range(len_j):
            rank[i][j] = float(rank[i][j])
    return rank

def copy2Darray(array2D):                        # copies 2D array 
    len_i = len(array2D)
    len_j = len(array2D[0])
    for i in range(len_i):
        for j in range(len_j):
            s[i][j] = array2D[i][j]
    return s

def copy1Darray(array1D):                        # copies 2D array 
    len_i = len(array1D)
    for i in range(len_i):
        s[i] = array1D[i]
    return s

instances = pd.read_csv('instance_Scaled.csv', sep=',',header=None)
labelsFeature = pd.read_csv('labelFeatures_Scaled.csv', sep=',',header=None)
orderings = pd.read_csv('ordering.csv', sep=',',header=None)
rankings = pd.read_csv('rankOrders.csv', sep=',',header=None)

labels = labelsFeature.as_matrix()[:,:].tolist()

scores = []
KtauScore = []

Ssplit = ShuffleSplit(n_splits=3, train_size=0.75, test_size=0.25, random_state=0)
for train_indices, test_indices in Ssplit.split(instances.as_matrix()[:,:]):
    
    train_orderings = orderings.as_matrix()[:,:][train_indices].tolist()
    test_orderings = orderings.as_matrix()[:,:][test_indices].tolist()

    train_instances_int = instances.as_matrix()[:,:][train_indices].tolist()
    train_instances = convertIntToDouble(train_instances_int)
    test_instances_int = instances.as_matrix()[:,:][test_indices].tolist()
    test_instances = convertIntToDouble(test_instances_int)
    
    train_rankings_int = rankings.as_matrix()[:,:][train_indices].tolist()
    test_rankings_int= rankings.as_matrix()[:,:][test_indices].tolist()
    test_rankings = convertIntToDouble(test_rankings_int)
    train_rankings = convertIntToDouble(train_rankings_int)
   
    double_class = gateway.jvm.double
    labels_features = gateway.new_array(double_class,len(labels),len(labels[0]))
    s = gateway.new_array(double_class,len(labels),len(labels[0]))
    labels_features = copy2Darray(labels)
    
    instances_tr = gateway.new_array(double_class,len(train_instances),len(train_instances[0]))
    s = gateway.new_array(double_class,len(train_instances),len(train_instances[0]))
    instances_tr = copy2Darray(train_instances)
    
    
    instances_te = gateway.new_array(double_class,len(test_instances),len(test_instances[0]))
    s = gateway.new_array(double_class,len(test_instances),len(test_instances[0]))
    instances_te = copy2Darray(test_instances)
    
    int_class = gateway.jvm.int
    ordering_tr = gateway.new_array(int_class,len(train_orderings),len(train_orderings[0]))
    s = gateway.new_array(int_class,len(train_orderings),len(train_orderings[0]))
    ordering_tr = copy2Darray(train_orderings)
    
    
    ranking_te = gateway.new_array(double_class,len(test_rankings),len(test_rankings[0]))
    s = gateway.new_array(double_class,len(test_rankings),len(test_rankings[0]))
    ranking_te = copy2Darray(test_rankings)
    
#    ranking_tr = gateway.new_array(double_class,len(train_rankings),len(train_rankings[0]))
#    s = gateway.new_array(double_class,len(train_rankings),len(train_rankings[0]))
#    ranking_tr = copy2Darray(train_rankings)
    
    
    prediction = gateway.new_array(double_class,len(ranking_te),len(ranking_te[0]))
    
    # BilinPL call
    prediction = java_object.trainAndTestBilinPL(labels_features, instances_tr, instances_te, ordering_tr, ranking_te)
    #print(java_object.trainAndTestBilinPL(labels_features, instances_tr, instances_te, ordering_tr, ranking_te))
    num = np.zeros((len(ranking_te),len(ranking_te[0])))
    
    for i in range(len(ranking_te)):
        for j in range(len(ranking_te[0])):
            num[i][j] = prediction[i][j]
        
    for i in range(len(ranking_te)):
        ktau, p_value = stats.kendalltau(test_rankings_int[i],num[i])
        scores.append(ktau)
        
    average_Score = sum(scores)/len(scores)
    KtauScore.append(average_Score)
    print("average score ",  average_Score)

final_average_Score = sum(KtauScore)/len(KtauScore)
print("Average Ktau score is :", final_average_Score)









