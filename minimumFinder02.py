# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:15:10 2013

@author: razibul
"""

from copy import deepcopy
nextHighest = lambda seq,x: min([(i-x,i) for i in seq if x<=i] or [(0,None)])[1]
nextLowest  = lambda seq,x: min([(x-i,i) for i in seq if x>=i] or [(0,None)])[1]
     
     
     
     
def get_closest_less(lst, target):
    ''' It takes array , target as an argument and returns nearest smaller'''
    nextLowest  = lambda lst,target: min([(target-i,i) for i in lst if target>=i] or [(0,None)])[1]
    return nextLowest



def indexsearch(index,usi):
    for i in range(len(usi)):
        if usi[i]==index:
            break
    return i
    
#    
#def get_closest_smaller(arrusi, target):
#    ''' It takes array , target as an argument and returns nearest smaller'''
#    arrusi.sort()
#    ui = None
#    previous = arrusi[0]
#    if (previous <= target):
#        for ndx in range(1, len(arrusi) - 1):
#            if arrusi[ndx] > target:
#                ui = previous
#                break
#            elif arrusi[ndx] == target:
#                ui = arrusi[ndx]
#            else:
#                previous = arrusi[ndx]
##    for index in range(len(usi)):
##        if usi[index]==ui:
##            break
#    return ui
    
def get_closest_smaller1(usi, target):
    ''' It takes array , target as an argument and returns nearest smaller'''
    if target == 0:
        index = 31
        return index
    arrusi = deepcopy(usi)
    arrusi.sort()
    ui = None
    previous = arrusi[0]
    if (previous <= target):
        for ndx in xrange(1, len(arrusi) - 1):
            if arrusi[ndx] > target:
                ui = previous
                break
            elif arrusi[ndx] == target:
                ui = arrusi[ndx]
            else:
                previous = arrusi[ndx]

#    for index in range(len(usi)):
#        if usi[index]==ui:
#            break
    return ui
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    