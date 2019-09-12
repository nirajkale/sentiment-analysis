#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:28:05 2019

@author: niraj
"""
import numpy as np

def get_shapes(*args):
    return [a.shape for a in args]

class DataChinker:
    
    def __init__(self):
        self.slices = None
        
    def __slices__(self,m,spans):
     last_count = 0
     slices = []
     for span in spans:
          temp = int(np.ceil(m*span))
          slices.append(slice(last_count, temp+last_count))
          last_count += temp
     return slices

    def prepare_slices(self, labels, spans=[0.75,0.1,0.15],is_one_hot = True):
        self.slices = []
        if is_one_hot:
            labels = np.argmax(labels ,axis=-1)
        _slices= []   
        for cid in np.unique(labels):
            indices = np.array(np.squeeze(np.argwhere(labels==cid)))
            if indices.size==1: #bug fix for class with only one value
                indices = indices.flatten()
            m_class = len(indices)
            _slices.append((cid, indices, self.__slices__(m_class, spans)))
        self.slices = _slices
        return _slices

    def slice_matrix(self, matrix, slices= None):
        slices = [slices if slices else self.slices][0]
        feature_sets = [[] for _ in range(len(slices[0][2]))]
        for cid, indices, slice_set in slices:
            for index,s in enumerate(slice_set):
                feature_sets[index].append( matrix[indices[s]] )
         #concatenate each set into a single vector
        for index,feature_set in enumerate(feature_sets):
            feature_sets[index] = np.concatenate(feature_set,axis=0)
        return feature_sets
    
    
if __name__ == '__main__':
    
    m = 100
    n = 5
    labels = np.eye(m,n)
    x1 = np.random.rand(m, 7)
    x2 = np.random.rand(m, 8)
    
    chinker = DataChinker()
    _ = chinker.prepare_slices(labels = labels)
    x1_sets = chinker.slice_matrix(feature = x1)
    x2_sets = chinker.slice_matrix(feature = x2)
    print(get_shapes( x1_sets))
    print(get_shapes( x2_sets))
    