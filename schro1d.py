#riley murray u1195514
#homework 5 problem 1

#useful imports
import numpy as np
from scipy import integrate
from scipy import sparse
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import time

### parameters ###
### parameters ###
x_init = -3
k = 50
sig = 0.25

hbar = 1
m = 1

coeff = (hbar/(2*m))
A = coeff*(1j)
C = 1/((sig**2)*np.pi)    #normalization constant

dx = 0.01                  #spatial grid spacing
x0 = -10.0
xf = 10.0
x = np.arange(x0,xf,dx)     #spatial grid points


dt = 0.001                       #temporal grid spacing
t0 = 0.0
tf = 0.2
t_eval = np.arange(t0,tf,dt)     #temporal grid points


figlength = 8
figheight = 8

xw = 1      #width of top hat
xc = 0      #center of top hat
xh = 1000   #height of top hat

### initial wave packet ###
gauss_term = C*np.exp((-(x-x_init)**2)/(2*sig**2))
imaginary_term = np.exp(1j*k*(x-x_init))

psi_init = gauss_term*imaginary_term
    
#print(psi_init)
#separate real and imaginary
yreal = np.real(psi_init)
#print(yreal)
yimag = np.imag(psi_init)
#print(yimag)

fig = plt.figure(figsize=(figlength,figheight))
fig.add_subplot(111)
plt.plot(x,yreal,color='g')
plt.xlabel('$x$')
plt.ylabel('$u(0,x)$')
plt.axis([x0,xf,-7,7])
plt.title('Initial Data - Real Component')
#plt.show()

fig = plt.figure(figsize=(figlength,figheight))
fig.add_subplot(111)
plt.plot(x,yimag,color='g')
plt.xlabel('$x$')
plt.ylabel('$u(0,x)$')
plt.axis([x0,xf,-7,7])
plt.title('Initial Data - Imaginary Component')
#plt.show()

### creating the energy barrier ###
barr = np.zeros(len(x))
for i in np.arange(0,len(x)):
    if ((x[i] > (xc - 0.5*xw)) and x[i] < (xc + 0.5*xw)):
        barr[i] = xh
        
fig = plt.figure(figsize=(figlength,figheight))
fig.add_subplot(111)
plt.plot(x,barr,color='g')
plt.xlabel('$x$')
plt.ylabel('$u(0,x)$')
plt.axis([x0,xf,-100,1100])
plt.title('Energy Barrier')
#plt.show()

### define finite difference laplace operator ###

DELSQ = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(len(x), len(x))) / dx**2

### define RHS of schrod equation ###
def dpsidt(t,psi):
    return A*(DELSQ.dot(psi))-((1j/hbar)*barr*psi)

### solve the ivp ###
tinit = time.time()
sol = integrate.solve_ivp(dpsidt,
                         t_span=[t0, tf],
                         y0=psi_init,
                         t_eval=t_eval,
                         method="RK23")
tfinal = time.time()
print("Time for solution = %5.2f seconds" % (tfinal-tinit) ) 
#print(len(sol.t))

### making the frames of the animation ###
for iframe in np.arange(0,len(sol.t)):

    output_filename = '%d.png' % iframe
    titlestring = "time = %5.3f" % t_eval[iframe]

    psi = sol.y[:,iframe]

    fig = plt.figure(figsize=(figlength, figheight))
    fig.add_subplot(111)
    plt.plot(x,psi)
    plt.xlabel('$x$')
    plt.ylabel('$psi(t,x)$')
    plt.title(titlestring)
    plt.axis([x0,xf,-5,5])
    plt.savefig(output_filename,format="png")
    plt.close(fig)

### stitching frames together to make animation ###
os.system("rm schro1d.mp4")
os.system("ffmpeg -i %d.png -vf scale=800x800 schro1d.mp4")
os.system("rm *.png")
