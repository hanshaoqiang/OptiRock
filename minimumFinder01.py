# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:15:10 2013

@author: razibul
"""
import numpy as np

nextHighest = lambda seq,x: min([(i-x,i) for i in seq if x<=i] or [(0,None)])[1]
nextLowest  = lambda seq,x: min([(x-i,i) for i in seq if x>=i] or [(0,None)])[1]
     
     
     
     
def get_closest_less(lst, target):
    ''' It takes array , target as an argument and returns nearest smaller'''
    lst.sort()
    ret_val = None
    previous = lst[0]
    if (previous <= target):
        for ndx in xrange(1, len(lst) - 1):
            if lst[ndx] > target:
                ret_val = previous
                break
            else:
                previous = lst[ndx]
    return ret_val



def indexsearch(index,usi):
    for i in range(len(usi)):
        if usi[i]==index:
            break
    return i



usi = np.array([99, 98, 92, 88, 80, 96, 64, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0])

print usi


print (nextLowest(usi,100))
print (indexsearch(nextLowest(usi,100),usi))