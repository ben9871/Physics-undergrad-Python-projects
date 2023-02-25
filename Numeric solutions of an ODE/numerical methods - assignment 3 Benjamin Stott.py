#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math as math
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
#import relevant libraries


# In[3]:


def f(t,x):
    return ((1+t)*x)+1-(3*t)+(t**2)
#define the ODE we want to examine


# In[38]:



x = np.linspace(-3, 3, 25)
t = np.linspace(0, 5, 25)
T, X = np.meshgrid(t, x)
#create a meshgrid in which a quiver plot will be made
dx = ((1+T)*X)+1-(3*T)+(T**2)
dt = np.ones((dx.shape))
#define dx and dt
plot2 = plt.figure()
plt.quiver(T, X, dt, dx, 
           color='black', 
           headlength=7,headwidth=9)
#quiver plot uses the meshgrid shape and dt,dx values to form arrows
plt.title('Quiver Plot, Single Colour')
plt.xlabel('t')
plt.ylabel('x')
#adds labels to plot
plt.show(plot2)
color = 'black'
lw = 1
plt.figure(2)
plt.title('Streamplot, single colour')
plt.streamplot(T,X,dt, dx, color=color, density=1., cmap='jet', arrowsize=1)
plt.xlabel('t')
plt.ylabel('x')
#creates a streamplot with labels for better visualisation of this critical point I speak of in my report


# In[39]:


def seuler(t,x,step):
    x_new = x + step*f(t,x)
    return x_new
#defines a simple euler function with variables x and t
def ieuler(t,x,step):
    x_new = x + 0.5*step*( f(t,x) + f(t+step, x + step*f(t, x)) )
    return x_new
#defines improved euler function
def rk(t,x,step):
    k1 = f(t,x)
    k2 = f(t + 0.5*step, x + 0.5*step*k1)
    k3 = f(t + 0.5*step, x + 0.5*step*k2)
    k4 = f(t + step, x + step*k3)
    x_new = x + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4) 
    return x_new
#defines 4th order runge kutta function

step = 0.04; start = 0.0; end = 5.0

x_zero = 0.0655

# n is the number of steps
n = int((end-start)/step)

# pre-allocate arrays
t = np.arange(start,end,step)
seul = np.zeros(n)


seul[0] = x_zero

# increment solutions
for i in range(1,n):
	seul[i] = seuler(t[i-1], seul[i-1], step)




plt.figure(2)
plt.streamplot(T,X,dt, dx, color=color, density=1, cmap='jet', arrowsize=1)
plt.title('Simple Euler method')
plt.plot(t, seul,color='red',linewidth=2
        )
plt.xlim(0,5)
plt.ylim(-3,3)
plt.xlabel('t')
plt.ylabel('x')
plt.figure(3)


plot2 = plt.figure()
plt.quiver(T, X, dt, dx, 
           color='black', 
           headlength=7,headwidth=9)

plt.title('Simple euler method')
plt.plot(t,seul,color='red',linewidth=2)
plt.xlim(0,5)
plt.ylim(-3,3)
plt.xlabel('t')
plt.ylabel('x')
plt.show(plot2)











#for j in x:
 #   for k in t:
  #      seul = seuler(k,j,step)
   #     seuler_.append(seul)


# In[30]:


step = 0.02; start = 0.0; end = 5.0
x_zero = 0.0655

# n is the number of steps
n = int((end-start)/step)

# pre-allocate arrays
t = np.arange(start,end,step)
seul = np.zeros(n)
ieul = np.zeros(n)
ruku = np.zeros(n)
real = np.zeros(n)

seul[0] = x_zero
ieul[0] = x_zero
ruku[0] = x_zero
real[0] = x_zero

# increment solutions
for i in range(1,n):
    seul[i] = seuler(t[i-1], seul[i-1], step)
    ieul[i] = ieuler(t[i-1], ieul[i-1], step)
    ruku[i] = rk(t[i-1], ruku[i-1], step)
    
    


# In[34]:


plt.figure(4)
plt.streamplot(T,X,dt, dx, color=color, density=1, cmap='jet', arrowsize=1)
plt.title('Improved Euler and Runge kutta methods')
plt.plot(t, ieul,color='green',linewidth=3,label='improved euler')
plt.plot(t,ruku,color='purple',linewidth=3,label='runge kutta')
plt.legend(loc='best')
plt.xlim(0,5)
plt.ylim(-3,3)
plt.xlabel('t')
plt.ylabel('x')
plt.figure(5)


plot2 = plt.figure()
plt.quiver(T, X, dt, dx, 
           color='black', 
           headlength=7,headwidth=9)

plt.title('Improved euler and runge kutta methods')
plt.plot(t,ieul,color='green',linewidth=3,label='improved euler')
plt.plot(t,ruku,color='purple',linewidth=3,label='runge kutta')
plt.legend(loc='best')
plt.xlim(0,5)
plt.ylim(-3,3)
plt.xlabel('t')
plt.ylabel('x')
plt.show(plot2)


# In[40]:


print('Name:Benjamin Stott, Student ID: 18336161')

