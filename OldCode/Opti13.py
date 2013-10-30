# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:01:24 2013

@author: razibul
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
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
#s = np.arange(30,-70,-1);
#s = np.repeat(0,100)
n =100;
h = 0.01;
dt = 0.004;
alpha = 0.01;
g = 9.81 ;
u = np.arange(4,dtype=np.float);
du = np.zeros(4,dtype=np.float);


def Function(u, du, s, dt):
    ''' This Function returns the second array of functions 
    which calculates the trajectory'''
    du[0]=u[2]; # distance in x direction
    du[1]=u[3]; # distance in y direction
    vel = np.math.sqrt(u[2]*u[2]+u[3]*u[3]);
    du[2] = s*u[2]/vel - u[2]*alpha*vel;
    du[3] = s*u[3]/vel -g - u[3]*alpha*vel;
    u[2]=du[0];
    u[3]=du[1];
    return u,du,vel;


def Trajectory(s):
    ''' This function calculates the trajectory of the rocket calling 
    Function for every point and plots the path'''
    u = np.arange(4,dtype=np.float);
    du = np.zeros(4,dtype=np.float);
    u[0] = 0.0;# position x
    u[1] = 0.0;# position y
    u[2] = 1.0;# velocity x
    u[3] = 1.0;# velocity y
    x= np.array([]);
    y= np.array([]);
    ua = np.array([u]);
    for i in range(n):
        u,du,vel=Function(u,du,s[i], dt);            
        for j in range(4):
            u[j]= u[j]+ du[j] * dt;
        x=np.append(x,u[0]) # X position of the rocket's trajectory
        y=np.append(y,u[1]) # Y position of the rocket's trajectory
        ua = np.append(ua,[u], axis = 0);
    
    return x,y,u,du,vel,ua ;

#To plot the trajectory of the projectile      
x,y,u,du,vel,uaopti = Trajectory(s);
'''
plt.figure(3)
plt.plot(x, y,'go', lw = '1.5');
plt.title('Trajectoy path before optimization \n');
plt.xlabel('distance, x')
plt.ylabel('distance, y')
plt.show();
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


def ObjFunc_u(ua,i):
    '''
    This function takes arrays of u vector and the iteration. 
    Returns the derivative of Objective function wrt. u vector.
    '''
    H_u = np.array([0.,0.,0.,0.]);
    u = ua[i];
    if i == n:
        H_u[0] = 2 * (u[0]-1);
        H_u[1] = 2*u[1];
        H_u[2] = 0;
        H_u[3]= 0;
    return H_u;
    

def Adjoint(s):
    ''' This function takes the thrust and returns the array of 
    adjoint vector'''
    x,y,u,du,vel,ua = Trajectory(s); # Length of the returns are variable
    I = np.identity(4);
    v = ObjFunc_u(ua,n);
    va = np.array([v]);
    for i in range((len(ua)-2),-1,-1):# it should be 
        part11 = (I + dt * Jacobian(ua[i],s[i]));
        part11 = np.asarray(part11);
        part12 = v;
        part1 = np.dot(part11.T,part12);
        part2 = ObjFunc_u(ua,(i));
        v = part1 + part2;
        va = np.append( [v],va, axis = 0);
    return va;


#va = Adjoint(s);
opti =  Adjoint(s)


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


def Objective(s):
    ''' This function recieves the positions Trajectory function 
    and using the last positional value calculates the objective functions
    value'''
    x,y,u,du,vel,ua = Trajectory(s);
    o = Fx(x[len(x)-1],y[len(y)-1]); # at the end
    return np.array([o]);

    
def Fx(x,y):
    ''' Obective function which returns the value wrt. x & y'''
    o = (x-1)**2 + y**2;
    return o;


def ObjFunc_s(s):
    '''    
    This function is returning the array of the gradient at iteration step.
    Case 1: It calculates by multiplying va(output of the adjoint function)
    dt and 
    '''
    Derivative = np.array([]); # array for the derivative at each iteration 
    x,y,u,du,vel,ua = Trajectory(s);
    va = Adjoint(s);
    for i in range(n):
        F_sx = F_s(ua[i]);
        mult = (np.dot(F_sx,va[i+1])) * dt;
        Derivative = np.append( Derivative,mult);
    return Derivative;



test = ObjFunc_s(s);

import scipy.optimize as so
list = [-10, 30];
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
x,y,u,du,vel,ua = Trajectory(res1.x)
plt.plot(x,y,'o')
plt.xlabel('Distance in x direction')
plt.ylabel('Distance in y direction')
plt.title('Trajectory after optimization \n \n')



#x0 = np.array([0])
#import scipy.optimize as so
#res = so.minimize(Objective, x0, method='BFGS',
#               options={'disp': True})


'''
# To check the output of the ObjFunc_s
DerivativeA = np.array([]);
ObjectiveA = np.array([]);
for i in range(40):
    x,y,u,du,vel,ua = Trajectory(i);
    Derivative = ObjFunc_s(i);
    DerivativeA = np.append(DerivativeA, Derivative)
    objective = Objective(i)
    ObjectiveA = np.append(ObjectiveA, objective);
plt.figure(1)
plt.plot(range(s), DerivativeA)
plt.xlabel('Thrust, s')
plt.ylabel('Gradient of Objective Function')
plt.title('Gradient of Objective Function vs Thrust \n \n')
plt.figure(2);
plt.plot(range(s), ObjectiveA);
plt.xlabel('Thrust, s');
plt.ylabel('Objective Function');
plt.title('Objective Function vs Thrust \n ');

#plt.plot(x,Derivative, lw = '1.5');
'''

'''
x0 = np.array([0])
import scipy.optimize as so
#res = so.minimize(ObjFunc_s, x0, method='BFGS',
#               options={'disp': True})
res1 = so.minimize(Objective, x0,jac = ObjFunc_s, method = 'BFGS', 
                    options = {'disp':True})
#res2 = so.fmin_bfgs(Objective,fprime = , x0)
'''