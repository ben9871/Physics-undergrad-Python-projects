
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae

# the fitting for the 2x2 jacobian case is simply obtained by changing parameters p here to 1 and 5, while specifying sampling points 0.1 and 0.5

#import relevant libraries
N= 5
#N value to create a 2N*2N jacobian
S = 2*N
# S sampling points
S_l = np.linspace(1, S, S)

#organising as indices
S_2 = np.linspace(0,S-1,S-1)
#organising as indices using pythonic indexing

S_3 =[]
for i in S_l:
    S_3.append(int(i))
S_3 =S_3[::2]
#every second value is taken in preparation to be used in loop where a value e.g 2 will be called as the beta parameter and i-1 will be the alpha parameter. allows us to generalise the gauss newton problem to N

N_l = np.linspace(1,N,N)
N_2 = np.linspace(0,N-1,N)


print(N_2)

#x = np.linspace(0.1,0.5, 2)
x = np.linspace(0, 1 , S)
#range of points over which to uniformly sample

#create a set of S uniformly spaced points to be used as samples
y=[] 
#empty list for y
for i in x:
    y.append( np.exp(-5*i))


#update y for each sampling point used

#p=[1,5]

p= []

for i in N_2:
    a = 1/N
    b = 1*3**(i)
    p.append(a)
    p.append(b)
p = np.array(p)
#print(len(p))    
#an empty list is made to contain p values where p means parameters aj, bj, aj+1, bj+1 .....etc
#a values are given 1/N values while b is given 1*3^(i) as specified

def gaussian(m, p):
    u = m
    r = -p[u-1] * np.exp(-p[u] * (x**2))
        
    return r
#a function that calculates a gaussian given a set of parameters p and the indices. this will be called on multiple times at once to give us the residual

def residual(p):
    ynew=y
    for i in S_3:
         res = gaussian(i, p)
         ynew +=  res
    return np.array(ynew)
#define a ynew array to be updated by subtracting one gaussian after another from it. the final result returns the residual. the loop opens by defining ynew as y to reset the array value with each call         

#print(residual(p))        


def jacobian_noT(m,p):
    j_i = -np.exp(-p[m] * (x**2)), x**2*p[m-1] * np.exp(-p[m] * (x**2))
    return list(j_i)


# define a function to calculate df(xi)/daj, df(xi)/dbj for individual parameter values across all x sampling points.
# combining the results of these for the N sets of parameter pairs allows us to construct the jacobian matrix quite easily


def jacobian(p):

    jacobian_full = []
    for i in S_3:
        j_it = jacobian_noT(i,p)
        jacobian_full.extend(j_it)
    return np.array(jacobian_full).T

# loops over the function to calculate jacobian entries for different parameter sets over all aj, bj and xi values. the jacobian entries from each loop are added to an empty list
# this list is then taken as a numpy array and transposed to return us the proper form

X_r = np.arange(x.min(),x.max(),0.01)

#define a range over which to plot models for explanatory purposes

y_fin=[]
for i in X_r:
    a = np.exp(-5*i)
    y_fin.append(a)

#create a matching y array corresponding to the function to which we are fitting
    
def guess(m, p, X_r):
    u = m
    k = p[u-1] * np.exp(-p[u] * (X_r**2))
        
    return k
# a function to calculate the model gaussian value at the points in the specified linspace for comparison purposes with the y=e^{-5x}


def guess_all(p):
    k_all=0
    for i in S_3:
         k = guess(i, p, X_r)
         k_all +=  k
    return k_all
    
#repeated for each set of parameters returning N gaussians




def guess_function(p):
    plt.plot(x, y, 'go', label="S sampling points")
    X_r = np.arange(x.min(),x.max(),0.01)
    Y = guess_all(p)
    plt.plot(X_r, Y, 'r-', label="$a_je^{-b_jx^2}$ iterations")
# a function to plot whichever guess we are making for any given set of p values on the same graph as the function to be fitted. makes it easier to implement within a loop

def function_to_be_fitted(x):
    plt.plot(x, np.exp(-5*x),label="$y = e^{-5x}$")


#the exponential function to which we are fitting with a function to graph it for arbitrary values of x sampling points

tol=1
#initial tolerance value just to get the simulation running
iterations = 1
while tol> 10**-6:
    iterations +=1
    lu, p_2 = la.lu_factor(jacobian(p))
    p = p + la.lu_solve((lu,p_2),-residual(p))/10
#solve for x in the equation ax=b where a is the jacobian and b is the negative of the residual. introduce a damping factor epsilon as  1/10 to stop the simulation from losing its way
    tol = la.norm(residual(p), 2)
    #print("The new parameters are {}".format(p))
    #print("The residual norm is", la.norm(residual(p), 2))
    #guess_function(p)
    #function_to_be_fitted(X_r)
    #plt.legend(loc="upper right")
    #plt.title("Gauss Newton fitting to $y=e^{-5x}$ using 6 gaussians, solving with the LU decomposition method",fontsize=8)
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.show()

#uncomment any of plotting and printing values above to see detailed info iteration by iteration

print("The final parameters for a tolerance of {} are {}".format(tol, p))
print("The residual norm is", la.norm(residual(p), 2))
print("the number of iterations required was {}".format(iterations))
#returns relevant parameters of this particular simulation if we were to compare it to another method(i.e QR decomposition) for the final p values
guess_function(p)
function_to_be_fitted(X_r)
plt.legend(loc="upper right")
plt.title("Gauss Newton fitting to $exponential$ using {} gaussians, solving with the LU decomposition method".format(N),fontsize=9)
plt.xlabel("x")
plt.ylabel("y")
plt.text(0.6, 0.6,("$MAD$ is =  {}".format(mae(guess_all(p), y_fin))), horizontalalignment='center',
     verticalalignment='center')
plt.text(0.6, 0.7,("iterations =  {}".format(iterations)), horizontalalignment='center',
     verticalalignment='center')

plt.show()


# N.B !!! mean absolute error of two curves was calculated simply using the mae function in the scikit.learn library for convenience
# graphing of the final iteration and all relevant values churned out.

#process is repeated for the QR process, resetting the values of the a_j, b_j parameters to their original guesses

p= []

for i in N_2:
    a = 1/N
    b = 1*3**(i)
    p.append(a)
    p.append(b)
p = np.array(p)
print(len(p))    

tol=1
#initial tolerance value just to get the simulation running
iterations = 0

while tol> 10**-6:
    iterations +=1
    Q, R = la.qr(jacobian(p), mode ='economic')
    c = np.dot(Q.T, -residual(p)) 
    #solve using the QR decomposition method this time around
    p = p +la.solve(R, c)/10
#solve for x in the equation ax=b where a is the jacobian and b is the negative of the residual. introduce a damping factor epsilon as  1/10 to stop the simulation from losing its way
    tol = la.norm(residual(p), 2)
    #print("The new parameters are {}".format(p))
    #print("The residual norm is", la.norm(residual(p), 2))
    #guess_function(p)
    #function_to_be_fitted(X_r)
    #plt.legend(loc="upper right")
    #plt.title("Gauss Newton fitting to $y=e^{-5x}$ using 6 gaussians, solving with the LU decomposition method",fontsize=8)
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.show()

print("The final parameters for a tolerance of {} are {}".format(tol, p))
print("The residual norm is", la.norm(residual(p), 2))
print("the number of iterations required was {}".format(iterations))
#returns relevant parameters of this particular simulation if we were to compare it to another method(i.e QR decomposition)
guess_function(p)
function_to_be_fitted(X_r)
plt.legend(loc="upper right")
plt.title("Gauss Newton fitting to $exponential$ using {} gaussians, solving with the QR decomposition method".format(N),fontsize=9)
plt.xlabel("x")
plt.ylabel("y")
plt.text(0.6,0.6,("$MAD$ is =  {}".format(mae(guess_all(p), y_fin ))), horizontalalignment='center',
     verticalalignment='center')
plt.text(0.6, 0.7,("iterations =  {}".format(iterations)), horizontalalignment='center',
     verticalalignment='center')

plt.show()





