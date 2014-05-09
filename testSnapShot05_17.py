# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:24:52 2013

@author: razibul
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from copy import deepcopy
#s = np.array([30,30,30,30,30,30,30,30,30,30,30,30,30,
#              30,30,30,30,30,30,30,30,30,30,30,30,30,
#              30,30,30,30,30,30,30,30,30,30,30,30,30,
#              30,30,30,30,30,30,30,30,30,30,30,30,0,0,
#              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
#              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]);
n =100
s = np.linspace(20,-20,n);
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


something=0
def FunctionForward(u, du, s):
    ''' This FunctionForward returns the second array of FunctionForwards
    which calculates the trajectory'''
    global something;
    something=something+1
    du[0]=u[2]; # distance in x direction
    du[1]=u[3]; # distance in y direction
    vel = np.math.sqrt(u[2]*u[2]+u[3]*u[3]);
    du[2] = s*u[2]/vel - u[2]*alpha*vel;
    du[3] = s*u[3]/vel -g - u[3]*alpha*vel;
    return du;
    
def Trajectory(s):
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

    return us,usi;
us,usi = Trajectory(s)


def SnapShot(s):
    global us, usi
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
#    us,usi = Trajectory(s)
    return


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


'''
def TrajectoryForAdjoint(i,s):
    u = np.arange(4,dtype=np.float);
    du = np.zeros(4,dtype=np.float);
    # find the best snapshot
    index = get_closest_smaller(usi,i)
    u= deepcopy(us[index])
    # iterate from there
    for i in range(usi[index],i,1):
        du = FunctionForward(u, du, s[i])
        for j in range(4):
            u[j]= u[j] + du[j] * dt
    return  u
'''
def indexsearch(index,usi):
    for i in range(len(usi)):
        if usi[i]==index:
            break
    return i

def TrajectoryForAdjoint(i,s):
    u = np.arange(4,dtype=np.float);
    du = np.zeros(4,dtype=np.float);
    # find the best snapshot
    nextLowest  = lambda seq,x: min([(x-i,i) for i in seq if x>=i] or [(0,None)])[1]
    index = indexsearch(nextLowest(usi,i),usi)
#    index = get_closest_smaller(usi,i)
#    print index
    u= deepcopy(us[index])
    # iterate from there
    for m in range(usi[index],i,1):
        du = FunctionForward(u, du, s[m])
        for j in range(4):
            u[j]= u[j] + du[j] * dt
        k= 0
        ii = m+1
        while  ii%2 == 0:
            k = k+1
            ii = ii/2
        usi[k] = m+1
        us[k] = deepcopy(u)
    
    return  u

    
'''
uas = np.zeros((n,4))
xs = np.zeros(n)
ys = np.zeros(n)
for i in range(n):
    u = TrajectoryForAdjoint(i,s)
    xs[i] = deepcopy(u[0])
    ys[i] = deepcopy(u[1])
    uas[i] = u
plt.plot(xs,ys)
'''

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

def Adjoint(s):
    ''' This function takes the thrust and returns the array of 
    adjoint vector'''
#    x,y,u,du,vel,ua = Trajectory(s); # Length of the returns are variable\
    u =TrajectoryForAdjoint(n,s)
    I = np.identity(4);
    v = ObjFunc_u(u,n);
    va = np.array([v]);
    for i in range((n-1),-1,-1):# it should be 
        u = TrajectoryForAdjoint(i,s)
        part11 = (I + dt * Jacobian(u,s[i]));
        part11 = np.asarray(part11);
        part12 = v;
        part1 = np.dot(part11.T,part12);
        part2 = ObjFunc_u(u,(i));
        v = part1 + part2;
        va = np.append( [v],va, axis = 0);
    return va;

#vatest = Adjoint();
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
    SnapShot(s)
    x = np.zeros(n)    
    y = np.zeros(n)    
#    for i in xrange(n):
    u = TrajectoryForAdjoint(n,s)
    x = u[0]
    y = u[1]
    o = Fx(x,y); # at the end
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
    va = Adjoint(s);
    for i in range(n):
        u = TrajectoryForAdjoint(i,s)
        F_sx = F_s(u);
        mult = (np.dot(F_sx,va[i+1])) * dt;
        Derivative = np.append( Derivative,mult);
    return Derivative;



def Optimize(x0):
    import scipy.optimize as so
    list = [0, 50];
    the_list=[]
    for i in range(100):
        the_list.append(list);
    
#    x0 = np.repeat(20,100);
#    x0 = np.linspace(20,-20,n);
                       
    res1 = so.minimize(Objective, x0,jac =ObjFunc_s, method = 'L-BFGS-B',
                       bounds=the_list);
    
    
    plt.figure(1)
    plt.plot(range(n),res1.x,lw=1.5)
    plt.xlabel('Iteraion')
    plt.ylabel('Optimized Thrust')
    plt.title('optimized s \n\n')
    plt.xlim(-2,110)
    plt.ylim(-2,32)
    #res1 = so.fmin_l_bfgs_b(Objective, x0, fprime=None)
    plt.figure(2)
    s = res1.x
    x = np.zeros(n)    
    y = np.zeros(n)    
    for i in xrange(n):
        u = TrajectoryForAdjoint(i,s)
        x[i] = u[0]
        y[i] = u[1]
    plt.plot(x,y, '.')
    plt.xlabel('Distance in x direction')
    plt.ylabel('Distance in y direction')
    plt.title('Trajectory after optimization \n \n')
    plt.ylim(-0.1,0.2)
    return
x0 = np.repeat(10,100);
test = Optimize(x0)
#ObjFunc_s(x0)
