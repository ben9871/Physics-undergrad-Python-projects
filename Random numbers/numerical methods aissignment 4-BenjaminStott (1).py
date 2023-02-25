#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as math
import scipy as scp
import scipy.stats as stats
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib as matplotlib
matplotlib.rc('axes',edgecolor='black')
#imports any librariers that may be useful
a=1
b=5
c=10
#for part 1, the averages we wish to take are 1,5,10 so its best to define these as a,b and c beforehand
#this avoids messiness


# In[2]:



def poisson(N,n):
    return (n ** N * np.exp(-n) /factorial(N))
#defines the poisson equation according to the lab manual
i= np.arange(0,50,1)
#defines the number of trials N
#plt.plot(i,poisson(i,a))
poissonv=np.vectorize(poisson)
plt.plot(i,poissonv(i,a),linewidth=3,label='distribution for <n>=1')
plt.plot(i,poissonv(i,b),linewidth=3,label='distribution for <n>=5')
plt.plot(i,poissonv(i,c),linewidth=3,label='distribution for <n>=10')
plt.legend(loc='best')
plt.xlabel('variable N')
plt.ylabel('Probability of occuring')
plt.title('poisson distributions for size N=50 varying averages')
plt.grid()
#adds labels and grid stylistics to plot
plt.bar(i,poissonv(i,a))
plt.bar(i,poissonv(i,b))
plt.bar(i,poissonv(i,c))
plt.gcf().set_facecolor('cadetblue')
#plot the poisson distribution  for varying averages in bar and line plot form


# In[3]:


poissonvalues=[]
totalnotsum=[]
meannotsum=[]
variancenotsum=[]

for i in range(0,50):
    poissonvalues.append(poisson(i,a))
    totalnotsum.append(poisson(i,a))
    meannotsum.append(i*poisson(i,a))
    variancenotsum.append((i**2)*poisson(i,a))
totalsum=np.sum(totalnotsum)
variance=np.sum(variancenotsum)
meansum=np.sum(meannotsum)
i=np.arange(0,5,1)
standard_deviation=np.sqrt(meansum)
#print(standard_deviation)
print(totalsum,'is the sum of probabilites',meansum,'is the expectation value',variance,'is the second moment')
#print(skewnotsum)
print(standard_deviation,'is the value of the standard deviation')


# In[4]:


#nothing much to see here. Just repeated the code for the above block for the next 2 averages
poissonvalues=[]
totalnotsum=[]
meannotsum=[]
variancenotsum=[]

for i in range(0,50):
    poissonvalues.append(poisson(i,b))
    totalnotsum.append(poisson(i,b))
    meannotsum.append(i*poisson(i,b))
    variancenotsum.append((i**2)*poisson(i,b))
#assigns lists of values to be summed for each variable asked for
totalsum=np.sum(totalnotsum)
variance=np.sum(variancenotsum)
meansum=np.sum(meannotsum)
i=np.arange(0,5,1)
standard_deviation=np.sqrt(meansum)
#print(standard_deviation)
print(totalsum,'is the sum of probabilites',meansum,'is the expectation value',variance,'is the second moment')
print(standard_deviation,'is the value of the standard deviation')
#prints off each variable asked for in the manual


# In[5]:


poissonvalues=[]
totalnotsum=[]
meannotsum=[]
variancenotsum=[]

for i in range(0,50):
    poissonvalues.append(poisson(i,c))
    totalnotsum.append(poisson(i,c))
    meannotsum.append(i*poisson(i,c))
    variancenotsum.append((i**2)*poisson(i,c))
totalsum=np.sum(totalnotsum)
variance=np.sum(variancenotsum)
meansum=np.sum(meannotsum)
i=np.arange(0,5,1)
standard_deviation=np.sqrt(meansum)
#print(standard_deviation)
print(totalsum,'is the sum of probabilites',meansum,'is the expectation value',variance,'is the second moment')
#print(skewnotsum)
print(standard_deviation,'is the value of the standard deviation')


# In[ ]:



N=50
#PART 2: SIMULATION OF DART THROWING

n_trials = (10, 100, 1000, 10000)
#defines the various amounts of trials at which we desire to run the simulation
Lregions = (100, 5)
#defines the number of regions in which the dart throwing takes place
for n_trial in n_trials:
    for L in Lregions:
        #simply loops over the values of trials and regions to return graphs for each
        H = np.zeros(N + 1)
        for _ in range(n_trial):
            b = np.zeros(L) #this is the array B mentioned in the manual
            for _ in range(N): b[np.random.randint(L)] += 1
 #obtain random integer values for the index of B which are then added as the numbers of darts in the regions
            for i in b: H[int(i)] += 1
                
        mean = sum(H * np.arange(0, N + 1)) / (n_trial * L)
        #calculates the mean of the values given
        H /= n_trial * L
        #normalises the distribution
        plt.grid(color='silver')
        plt.semilogy(np.arange(0, N + 1), poisson(np.arange(0, N + 1), mean), 'd', label=r'$P(n)$',color='red')
        plt.semilogy(np.arange(0, N + 1), H,'o', label=r'$P_{sim}(n)$',color='black')
        #plots the analytic and simulated distributions on semilog axes
        plt.xlabel('N')
        plt.ylabel('P(n)')
        plt.legend()
        plt.title(fr'normalised distribution $n_{{trials}}={n_trial},L={L}$')
        plt.gcf().set_facecolor('silver')
        #adds some stylisation and titles to the plot
        plt.show()
       
        print('smallest value of P(n)',min(H[H > 0]))  
        #denotes the minimum value of each distribution


# In[ ]:


print('Name:Benjamin Stott, Student ID: 18336161')


# In[ ]:




