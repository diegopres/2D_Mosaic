# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:27:14 2019

@author: uidv7259
"""
#import random
import numpy as np


class K_means():
    def __init__(self, n_clusters = 3, iterations = 10):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.labels_ = []
        self.cluster_centers_ = np.zeros((self.n_clusters, 3))
        self.mov = 0
        self.comp = 0
        
    def dist(self, a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)
                
    def fit(self, arr):
        size_arr = len(arr)
        self.labels_ = np.zeros(size_arr)
        labels = np.zeros(size_arr)
        cluster_centers = np.zeros((self.n_clusters, 3))
        sum_mean = np.zeros((self.n_clusters, 3))
        total_sums = [0 for i in range(self.n_clusters)]
        min_variance = None
        total_mean_iterations = 50
        distances_2clusters = [[[] for i in range(self.n_clusters)] for j in range(self.iterations)]
        self.mov = 0
        self.comp = 0
        
        for it in range(self.iterations):
            variance = np.zeros(self.n_clusters)
            no_change = False
            mean_iterations = 0
            
            for x in range(self.n_clusters):
                n = np.random.randint(0, size_arr - 1)
                cluster_centers[x] = arr[n]
            
            while no_change == False and mean_iterations < total_mean_iterations:
                no_change = True
                mean_iterations += 1
                for i in range(size_arr):
                    distances = self.dist(arr[i], cluster_centers)
                    cluster = np.argmin(distances)
                    self.mov += 1 #Count movements depending on number of clusters
                    
                    self.comp += 1 #Comparison to check change of clusters
                    if labels[i] != cluster:
                        no_change = False
                    labels[i] = cluster
                    distances_2clusters[it][cluster].append(distances[cluster])
                    
                    rd, gd, bd = arr[i]
                    sum_mean[cluster, 0] += rd 
                    sum_mean[cluster, 1] += gd
                    sum_mean[cluster, 2] += bd
                    total_sums[cluster] += 1
                    self.mov += 1 #Count movements depending on array size
                    
                for n in range(self.n_clusters):
                    div = total_sums[n] if total_sums[n] > 1 else 1
                    sum_mean[n, 0] /= div
                    sum_mean[n, 1] /= div
                    sum_mean[n, 2] /= div
                    cluster_centers[n] = sum_mean[n, :]
                    sum_mean[n, 0] = 0
                    sum_mean[n, 1] = 0
                    sum_mean[n, 2] = 0
                    total_sums[n] = 0
                    self.mov += 1 #Count movements depending on number of clusters
            
            for j in range(self.n_clusters):
                variance[j] = np.var(distances_2clusters[it][j])
                self.mov += 1 #Count movements depending on number of clusters
                
            var_of_var = np.var(variance)
            
            self.comp += 1 #Comparison to check variance of clusters
            if min_variance == None:
                self.labels_ = labels
                self.cluster_centers_ = cluster_centers
                min_variance = var_of_var
                self.mov += 1 #Count movements when first min_variance
            elif var_of_var < min_variance:
                self.labels_ = labels
                self.cluster_centers_ = cluster_centers
                min_variance = var_of_var
                self.mov += 1 #Count movements when new min_variance
            

        
            
        
            
            
        
        
        
        
                    
        