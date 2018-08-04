from __future__ import division

'''  Potential field initiated in cartesian coordinates '''

import numpy as np
import scipy as sc
from scipy.integrate import dblquad
from mpi4py import MPI

'''
Need to choose parameters. 'Gam' is the circulation. 
C1,C2,C3 are the parameters of the kernel used for the distribution of
singular vortex rings. (Gaussian-like: C1*a*exp(-C2*a**2-C3*z0**2))
z1,z2,a1,a2, are the limits of integration.
Xi,Xf,Yi,Yf,Zi,Zf, are the boundaries of the domain (here a cube of length 20).
epsabs, epsrel : absolute and relative precisions for dblquad
'''
Gam = 1
C1 = np.sqrt(2)
C2 = 1/4
C3 = 1/4
z1 = -4
z2 = 4
a1 = 1e-5
a2 = 5
Xi = -10.0 
Xf = 10
Yi = -10
Yf = 10.0
Zi = -10
Zf = 10.0
epsabs = 1e-4
epsrel = 1e-4
'''
Functions definition below. (integrand for potential Psi)
The integrand is formed of the singular line ring potential
and the kernel.
'''

def integrand_Psi_CART(x,y,z,a,z0):
    k_CART = (2*np.sqrt(a*np.sqrt(x**2+y**2)))/(np.sqrt(a**2+x**2+y**2+(z-z0)**2+2*a*np.sqrt(x**2+y**2)))
    
    if k_CART.any() < 1e-3:
        EllipticK_CART = sc.special.ellipkm1((1-k_CART)**2)
        EllipticE_CART = sc.special.ellipe(k_CART**2)
    else:
        EllipticK_CART = sc.special.ellipk(k_CART**2)
        EllipticE_CART = sc.special.ellipe(k_CART**2)
        
    ret_CART = (C1*a*np.exp(-C2*a**2-C3*z0**2))*(Gam*a/np.pi)*(1/np.sqrt(a**2+x**2+y**2+(z-z0)**2+2*a*np.sqrt(x**2+y**2)))*(((2-k_CART**2)*EllipticK_CART-2*EllipticE_CART)/(k_CART**2))
    return(ret_CART)
  
''' 
Processors
'''

nprocx = 4
nprocy = 4
nprocz = 2
    
        
'''
Preparing domain (number of points in mesh for each direction)
'''
nb_pt_X = 256
nb_pt_Y = 256 
nb_pt_Z = 256

dX = (Xf-Xi)/(nb_pt_X-1)  
dY = (Yf-Yi)/(nb_pt_Y-1)
dZ = (Zf-Zi)/(nb_pt_Z-1)


#Add boundary (3 on each face, 2 faces for each direction = 6)
nb_pt_X += 6
nb_pt_Y += 6
nb_pt_Z += 6

#linX_negative = np.linspace(-Xf-3*dX,-Xi,nb_pt_X/2)
#linX_positive = np.linspace(Xi,Xf+3*dX,nb_pt_X/2)
#linX = np.concatenate((linX_negative,linX_positive))
#linY = np.linspace(Yi-3*dY,Yf+3*dY,nb_pt_Y)
#linZ = np.linspace(Zi-3*dZ,Zf+3*dZ,nb_pt_Z)

linX = np.linspace(Xi-3*dX,Xf+3*dX,nb_pt_X)
linY = np.linspace(Yi-3*dY,Yf+3*dY,nb_pt_Y)
linZ = np.linspace(Zi-2*dZ-dZ/2,Zf+2*dZ+dZ/2,nb_pt_Z)


np.save('LinX',linX)
np.save('LinY',linY)
np.save('LinZ',linZ)


''' 
Proc. struct and partial arrays
'''
comm = MPI.COMM_WORLD
proc = comm.Get_rank()
nproc = comm.Get_size()



###############################################
###     1/8'th of the whole space           ###
###############################################

linX_pos = linX[linX>=0]
linY_pos = linY[linY>=0]
linZ_pos = linZ[linZ>=0]

Psi_partial_temp = np.zeros([nb_pt_X//2,nb_pt_Y//2,nb_pt_Z//2])[:,:,proc::nproc]
Psi_X_partial_temp = np.zeros_like(Psi_partial_temp)
Psi_Y_partial_temp = np.zeros_like(Psi_partial_temp)


'''
For proc. 0 -> full arrays
'''
if proc == 0:
    Psi_X_temp = np.zeros([nb_pt_X//2,nb_pt_Y//2,nb_pt_Z//2])
    Psi_Y_temp = np.zeros_like(Psi_X_temp)


'''
Computation only on positive quadrant
'''

for k, Z in enumerate(linZ_pos[proc::nproc]):
    for i, X in enumerate(linX_pos):
        for j, Y in enumerate(linY_pos):
            func_int = lambda z0,a: integrand_Psi_CART(X,Y,Z,a,z0)
            low_bd_z = lambda z0: z1
            up_bd_z= lambda z0: z2
            Psi_partial_temp[i,j,k] = dblquad(func_int,a1,a2,low_bd_z,up_bd_z,epsabs=epsabs,epsrel=epsrel)[0]
            Psi_X_partial_temp[i,j,k] = Psi_partial_temp[i,j,k]*(-np.sin(np.arctan2(Y,X)))
            Psi_Y_partial_temp[i,j,k] = Psi_partial_temp[i,j,k]*(np.cos(np.arctan2(Y,X)))



# Communicate results to proc 0.
if proc == 0:
    Psi_X_temp[:, :, ::nproc] = Psi_X_partial_temp
    Psi_Y_temp[:, :, ::nproc] = Psi_Y_partial_temp
    for p in np.arange(1, nproc):
        Psi_X_temp[:, :, p::nproc] = comm.recv(source=p, tag=0)
        Psi_Y_temp[:, :, p::nproc] = comm.recv(source=p, tag=1)
else:
    comm.send(Psi_X_partial_temp, dest=0, tag=0)
    comm.send(Psi_Y_partial_temp, dest=0, tag=1)

'''
Mirror the fields
'''

if proc == 0:
    Psi_X = np.zeros([nb_pt_X,nb_pt_Y,nb_pt_Z])
    Psi_Y = np.zeros_like(Psi_X)
    
    #Fill in the back upper right quadrant X(+),Y(+),Z(+)
    Psi_X[nb_pt_X//2:,nb_pt_Y//2:,nb_pt_Z//2:] = Psi_X_temp
    Psi_Y[nb_pt_X//2:,nb_pt_Y//2:,nb_pt_Z//2:] = Psi_Y_temp
    
    #Fill in the back upper left quadrant X(-),Y(+),Z(+)
    Psi_X[:nb_pt_X//2,nb_pt_Y//2:,nb_pt_Z//2:] = Psi_X_temp[::-1,:,:]
    Psi_Y[:nb_pt_X//2,nb_pt_Y//2:,nb_pt_Z//2:] = -Psi_Y_temp[::-1,:,:]
    
        #If problem at 0
        #Psi_X[:nb_pt_X//2,nb_pt_Y//2:,nb_pt_Z//2:] = Psi_X_temp[::-1,:,:][:-1,:,:]
        #Psi_Y[:nb_pt_X//2,nb_pt_Y//2:,nb_pt_Z//2:] = Psi_Y_temp[::-1,:,:][:-1,:,:]
    
    
    #Fill in the front upper right quadrant X(+),Y(-),Z(+)
    Psi_X[nb_pt_X//2:,:nb_pt_Y//2,nb_pt_Z//2:] = -Psi_X_temp[:,::-1,:]
    Psi_Y[nb_pt_X//2:,:nb_pt_Y//2,nb_pt_Z//2:] = Psi_Y_temp[:,::-1,:]
    
    #Fill in the front upper left quadrant X(-),Y(-),Z(+)
    Psi_X[:nb_pt_X//2,:nb_pt_Y//2,nb_pt_Z//2:] = -Psi_X_temp[::-1,::-1,:]
    Psi_Y[:nb_pt_X//2,:nb_pt_Y//2,nb_pt_Z//2:] = -Psi_Y_temp[::-1,::-1,:]
    
    #Fill in the back lower right quadrant X(+),Y(+),Z(-)
    Psi_X[nb_pt_X//2:,nb_pt_Y//2:,:nb_pt_Z//2] = Psi_X_temp[:,:,::-1]
    Psi_Y[nb_pt_X//2:,nb_pt_Y//2:,:nb_pt_Z//2] = Psi_Y_temp[:,:,::-1]
    
    #Fill in the back lower left quadrant X(-),Y(+),Z(-)
    Psi_X[:nb_pt_X//2,nb_pt_Y//2:,:nb_pt_Z//2] = Psi_X_temp[::-1,:,::-1]
    Psi_Y[:nb_pt_X//2,nb_pt_Y//2:,:nb_pt_Z//2] = -Psi_Y_temp[::-1,:,::-1]
    
    #Fill in the front lower right quadrant X(+),Y(-),Z(-)
    Psi_X[nb_pt_X//2:,:nb_pt_Y//2,:nb_pt_Z//2] = -Psi_X_temp[:,::-1,::-1]
    Psi_Y[nb_pt_X//2:,:nb_pt_Y//2,:nb_pt_Z//2] = Psi_Y_temp[:,::-1,::-1]
    
    #Fill in the front lower left quadrant X(-),Y(-),Z(-)
    Psi_X[:nb_pt_X//2,:nb_pt_Y//2,:nb_pt_Z//2] = -Psi_X_temp[::-1,::-1,::-1]
    Psi_Y[:nb_pt_X//2,:nb_pt_Y//2,:nb_pt_Z//2] = -Psi_Y_temp[::-1,::-1,::-1]
    
    
'''
Write the Pencil Code file 
'''

if proc == 0:
    #PencilCode format.
    Psi_final = np.zeros([3,nb_pt_X,nb_pt_Y,nb_pt_Z], dtype=np.float32)
    Psi_final[0, ...] = Psi_X #X component
    Psi_final[1, ...] = Psi_Y #Y component
    #No Z component
    
    # Write the output VAR file
    iproc = 0
    n_X = nb_pt_X-6
    n_Y = nb_pt_Y-6
    n_Z = nb_pt_Z-6
    for iz in range(nprocz):
        for iy in range(nprocy):
            for ix in range(nprocx):
                Psi_partial = Psi_final[:, ix*int(n_X/nprocx):(ix+1)*int(n_X/nprocx)+6,
                                iy*int(n_Y/nprocy):(iy+1)*int(n_Y/nprocy)+6,
                                iz*int(n_Z/nprocz):(iz+1)*int(n_Z/nprocz)+6]
                np.swapaxes(Psi_partial, 1, 3).tofile('out{0}'.format(iproc))
                iproc += 1

    np.save('Psi_final', Psi_final)
 
    #np.save('Psi_temp',Psi_partial)
 
'''
END
'''
