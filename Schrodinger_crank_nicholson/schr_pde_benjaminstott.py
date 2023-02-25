
import matplotlib.pyplot as plt
import numpy as np
import math



nx = 400
# Number of space points
nt=2100 # Number of time steps
psi=np.zeros([nt,nx+2],dtype = 'complex_')
#initial matrix, dtype complex necessary as python cannot read otherwise
#print(psi)
dx=0.1
dt=0.001
#timestep and xstep
k, Sig, x_0 = 2 , 1, 20
#relevant constants

xax=np.arange(nx)*dx # x-axis
t_axis=np.arange(nt)*dt
#print(xax)
# init value
psi[0,1:(nx+1)]= (1/(Sig**0.5*np.pi**0.25))*np.exp(-(xax-x_0)**2/2/Sig**2)*np.exp(1j*k*xax)
psi[0,0]=psi[0,nx]
psi[0,1]=psi[0,nx+1]
#periodic b.c. for two points as opposed to 4 in the soliton case


for b in range(1,(nx+1)):
    psi[1,b] = psi[0,b] + 1j * dt/dx**2 * (psi[0,b+1] - 2*psi[0,b] + psi[0,b-1])

#one timestep iteration of the difference scheme in part(a) before moving onto the more accurate scheme

psi[1,0]=psi[1,nx]
psi[1,1]=psi[1,nx+1]
#periodic b.c.


for i in range(2,nt): # time loop
    for b in range(1,(nx+1)): # space loop
        
        psi[i,b] = psi[i-2,b] + 2j * dt/dx**2 * (psi[i-1,b+1] - 2*psi[i-1,b] + psi[i-1,b-1])
   
    psi[i,0]=psi[i,nx]
    psi[i,1]=psi[i,nx+1]
    #periodic b.c.
    

#t=0 steps (initial condition)
plt.figure(1)
plt.plot(xax, np.abs(psi[0,2:(nx+2)])**2, label = "t=0")
#plt.show()
#t=1000 steps    
plt.plot(xax,np.abs(psi[1000,2:(nx+2)])**2, label ="t=1")
#lt.show()
#t=2000 steps 
plt.plot(xax,np.abs(psi[2000,2:(nx+2)])**2, label="t=2") 
plt.xlabel("x")
plt.ylabel("$|\psi|^2$ (probability density)")
plt.title("$|\psi|^2$ for t = 0 , 1 and 2")
plt.legend(loc="best")
plt.show()
#t=9000 steps to show nonphysical behaviour
#plt.plot(xax,np.abs(psi[9000,2:(nx+2)])**2)

integrate = []

def trapz(f):
   
    y = f
    y_r = y[1:] # right endpoints
    y_l = y[:-1] # left endpoints
    dx = 0.1
    T = (dx/2) * np.sum(y_r + y_l)
    
    return T


for i in range(2100):
    f =  np.abs(psi[i,2:(nx+2)])**2
    a = trapz(f)
    integrate.append(a)

plt.figure(2)
plt.plot( t_axis, integrate, markersize="0.1")
plt.title("$\int_0^{nt}|\psi|^2dx$ vs t")
plt.xlabel("t")
plt.ylabel("$\int_0^{nt}|\psi|^2dx$")
plt.show()




