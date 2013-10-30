# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:24:52 2013

@author: razibul
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from copy import deepcopy
#s = np.array([0,0,0,0,0,0,0,0,0,0,10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,40,40,40,40,30,30,30,30,30,30,30,30,30,30,20,20,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,-10,-10,-10,-10,-10,-10,-10]);
#s = np.array([0,0,0,0,0,0,20,20,20,20,10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,40,40,40,40,30,30,30,30,30,30,30,30,30,30,20,20,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,-10,-10,-10,-10,-10,-10,-10]);
#s = np.array([0,0,0,0,0,0,20,20,20,20,10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,40,40,40,40,30,30,30,30,30,30,30,30,30,30,20,20,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,100,100,100,100,100,100,100]);
#s = np.array([50,50,50,50,50,50,20,20,20,20,10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,40,40,40,40,30,30,30,30,30,30,30,30,30,30,20,20,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,0,0,0,0,0,0,0,0,0,0,0,100,100,100,100,100,100,100]);
s = np.array([30,30,30,30,30,30,30,30,30,30,30,30,30,
              30,30,30,30,30,30,30,30,30,30,30,30,30,
              30,30,30,30,30,30,30,30,30,30,30,30,30,
              30,30,30,30,30,30,30,30,30,30,30,30,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]);
n =100
#s = np.linspace(20,-20,n);
#s = 30
h = 0.01
dt = 0.004
alpha = 0.01
g = 9.81
u = np.arange(4,dtype=np.float)
du = np.zeros(4,dtype=np.float)

us = np.zeros((32,4));
usi = np.zeros(32, dtype = np.int)
for i in range(32):
    usi[i]=-1;

u[0] = 0.0;# position x
u[1] = 0.0;# position y
u[2] = 1.0;# velocity x
u[3] = 1.0;# velocity y
usi[31] = 0
us[31] = deepcopy(u)


def FunctionForward(u, du, s):
    ''' This FunctionForward returns the second array of FunctionForwards
    which calculates the trajectory'''
    du[0]=u[2]; # distance in x direction
    du[1]=u[3]; # distance in y direction
    vel = np.math.sqrt(u[2]*u[2]+u[3]*u[3]);
    du[2] = s*u[2]/vel - u[2]*alpha*vel;
    du[3] = s*u[3]/vel -g - u[3]*alpha*vel;
    return du;
    
def Trajectory():
    ''' This function calculates the trajectory of the rocket calling
    Function for every point and plots the path'''
    u = np.arange(4,dtype=np.float);
    du = np.zeros(4,dtype=np.float);
    u[0] = 0.0;# position x
    u[1] = 0.0;# position y
    u[2] = 1.0;# velocity x
    u[3] = 1.0;# velocity y
    us = np.zeros((32,4));
    usi = np.zeros(32, dtype = np.int)
    for i in range(32):
        usi[i]=-1;
    ii = 0
    if ii == 0:
        k=31
    else :
        while  ii%2 == 0:
            k = k+1
            ii = ii/2
    usi[k] = 0
    us[k] = deepcopy(u)
    
    for i in range(n):
        du=FunctionForward(u,du,s[i]);
        for j in range(4):
            u[j]= u[j]+ du[j] * dt;
        k= 0
        ii = i+1
        if ii == 0:
            k=31
        else :
            while  ii%2 == 0:
                k = k+1
                ii = ii/2
        usi[k] = i+1
        us[k] = deepcopy(u)
#    us[len(us)-1] = ua[len(ua)-1]
#    usi[len(usi)-1] = len(ua)-1
#    usi[len(usi)-1]= n
#    us[len(usi)-1]=deepcopy(ua[n])
#    print kCheck
    return us,usi;
us,usi = Trajectory()
#print usi
'''
#To check the snapshot with the forward calculation
us,usi = Trajectory(s)
#print usi,'\n',us
nonzero = np.nonzero(usi)
for i in nonzero:
    diff = (us[i]-ua[usi[i]]);
#    print us[i];us
print diff


us,usi = Trajectory(s)
nonzero = np.nonzero(usi)
test = nonzero[0]
testus= np.zeros((len(test),4))
testusi= np.zeros(len(test))

for i in range(len(test)):    
    testusi[i] = usi[test[i]]    
    
print testusi
print testus
'''

def get_closest_smaller(usi, target):
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
    for index in range(len(usi)):
        if usi[index]==ui:
            break
    return index



def TrajectoryForAdjoint(i):
    u = np.arange(4,dtype=np.float);
    du = np.zeros(4,dtype=np.float);
    # find the best snapshot
    index = get_closest_smaller(usi,i)
#    print index
    u= deepcopy(us[index])
#    print str(index),u
    # iterate from there
    for i in range(usi[index],i,1):
        du = FunctionForward(u, du, s[i])
        for j in range(4):
            u[j]= u[j] + du[j] * dt
#    usi[0]= i
#    us[0] = u
    return  u
    

uas = np.zeros((n,4))
xs = np.zeros(n)
ys = np.zeros(n)
for i in range(n):
    u = TrajectoryForAdjoint(i)
    xs[i] = deepcopy(u[0])
    ys[i] = deepcopy(u[1])
    uas[i] = u
plt.plot(xs,ys)

def JacobianFunction():
    s = Symbol('s');
    u1 = Symbol('u1');
    u2 = Symbol('u2');
    u3 = Symbol('u3');
    u4 = Symbol('u4');
    alphasy = Symbol('alphasy');
    F1 = u3;
    F2 = u4;
    F3 = s*u3*(u3**2+u4**2)**(-0.5)-u3*alphasy*(u3**2+u4**2)**(0.5);
    F4 = s*u4*(u3**2+u4**2)**(-0.5)-u4*alphasy*(u3**2+u4**2)**(0.5)-g;
    F = Matrix([F1,F2,F3,F4]);
    u = Matrix([u1,u2,u3,u4]);
    J = F.jacobian(u);
    lambdifiedJacobian= lambdify((u1,u2,u3,u4,alphasy,s),J);
    return lambdifiedJacobian;

lambdifiedJacobian = JacobianFunction();
#print lambdaJacobian(u[0],u[1],u[2],u[3],alpha,1);


def Jacobian(uprime,sprime):  
    ''' this function produces the Jacobian matrix and returns
    jacobian matrix provided u and s'''
    a = lambdifiedJacobian(uprime[0],uprime[1],
                           uprime[2],uprime[3],alpha,sprime);
    return a;


def ObjFunc_u(u,i):
    '''
    This function takes arrays of u vector and the iteration. 
    Returns the derivative of Objective function wrt. u vector.
    '''
    H_u = np.array([0.,0.,0.,0.]);
    u = deepcopy(u)
    if i == n:
        H_u[0] = 2 * (u[0]-1);
        H_u[1] = 2*u[1];
        H_u[2] = 0;
        H_u[3]= 0;
    return H_u;

def Adjoint():
    ''' This function takes the thrust and returns the array of 
    adjoint vector'''
#    x,y,u,du,vel,ua = Trajectory(s); # Length of the returns are variable\
    u =TrajectoryForAdjoint(n)
    I = np.identity(4);
    v = ObjFunc_u(u,n);
    va = np.array([v]);
    for i in range((n-1),-1,-1):# it should be 
        u = TrajectoryForAdjoint(i)
        part11 = (I + dt * Jacobian(u,s[i]));
        part11 = np.asarray(part11);
        part12 = v;
        part1 = np.dot(part11.T,part12);
        part2 = ObjFunc_u(u,(i));
        v = part1 + part2;
        va = np.append( [v],va, axis = 0);
    return va;

vatest = Adjoint();
##########################################
def F_s(u):
    ''' It takes the vector of u with four component and 
    returns the derivative of the F(u_i,s) wrt "s" '''
#    print  'u',  u;
    F_sx = np.arange(4,dtype=np.float64);
    F_sx[0] = 0;
    F_sx[1] = 0;
    F_sx[2] = u[2]* (u[2]**2+u[3]**2)**(-0.5);
    F_sx[3] = u[3]* (u[2]**2+u[3]**2)**(-0.5);
    return F_sx;

def Fx(x,y):
    ''' Obective function which returns the value wrt. x & y'''
    o = (x-1)**2 + y**2;
    return o;


def Objective(s):
    ''' This function recieves the positions Trajectory function 
    and using the last positional value calculates the objective functions
    value'''
#    x,y,u,du,vel,ua = Trajectory(s);
    x = np.zeros(n)    
    y = np.zeros(n)    
    for i in xrange(n):
        u = TrajectoryForAdjoint(i)
        x[i] = u[0]
        y[i] = u[1]
    o = Fx(x[len(x)-1],y[len(y)-1]); # at the end
    return np.array([o]);
#test = Objective()
    



def ObjFunc_s(s):
    '''    
    This function is returning the array of the gradient at iteration step.
    Case 1: It calculates by multiplying va(output of the adjoint function)
    dt and 
    '''
    Derivative = np.array([]);
#    x,y,u,du,vel,ua = Trajectory();
    va = Adjoint();
    for i in range(n):
        u = TrajectoryForAdjoint(i)
        F_sx = F_s(u);
        mult = (np.dot(F_sx,va[i+1])) * dt;
        Derivative = np.append( Derivative,mult);
    return Derivative;

'''
test = ObjFunc_s(s);

import scipy.optimize as so
list = [0, 10];
the_list=[]
for i in range(100):
    the_list.append(list);
bound1 = np.array(the_list);

x0 = np.repeat(10,100);
#res1 = so.minimize(Objective, x0,jac = ObjFunc_s, method = 'L-BFGS-B',
#                   bounds=the_list, options = {'disp':True});
                   
res1 = so.minimize(Objective, x0,jac = ObjFunc_s, method = 'L-BFGS-B',
                   bounds=the_list);

plt.figure(1)
plt.plot(res1.x)
plt.title('optimized s')
#res1 = so.fmin_l_bfgs_b(Objective, x0, fprime=None)
plt.figure(2)
s = res1.x
x = np.zeros(n)    
y = np.zeros(n)    
for i in xrange(n):
    u = TrajectoryForAdjoint(i)
    x[i] = u[0]
    y[i] = u[1]
plt.plot(x,y)
plt.xlabel('Distance in x direction')
plt.ylabel('Distance in y direction')
plt.title('Trajectory after optimization \n \n')
'''
#
##for i in range(77,90,1):
##    print 'i is',i
##    TrajectoryI(i)
##TrajectoryI(98)
## This is to check whether the snapshot mechanism works
##x,y,u,du,vel,ua, dua = Trajectory(s);
#'''
#xs = np.zeros(n)
#ys = np.zeros(n)
#for i in range(n):
#
#    u = TrajectoryForAdjoint(i)
#    xs[i] = deepcopy(u[0])
#    ys[i] = deepcopy(u[1])
#plt.plot(xs,ys)
#'''
#
#
#
#
#
#
