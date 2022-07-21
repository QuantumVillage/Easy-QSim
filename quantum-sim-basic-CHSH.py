import numpy as np
from numpy import ndarray

np.set_printoptions(edgeitems=30, linewidth=100000)

class simarr(ndarray):
    @property
    def H(self):
        return self.conj().T

u = np.array([[1+0j],[0+0j]]).view(simarr) # up qbit 
d = np.array([[0+0j],[1+0j]]).view(simarr) # down qbit
i2 = np.eye(2)

k = np.kron
op0 = k(u, u.H)
op1 = k(d, d.H)
X = np.array([[0,1],[1,0]]).view(simarr)
sX = np.array([[(1+1j)/2,(1-1j)/2],
              [(1-1j)/2,(1+1j)/2]]).view(simarr)
H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)]]).view(simarr)
def Rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]]).view(simarr)
def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]]).view(simarr)


"""
Circuit Diagram - CHSH:
 s1 s2 s3 s4 s5 s6 s7
-H--c--Rx-sX-sX-
----X-----|--|--
-H--------c--|--
-H-----------c--
"""

s1 = k(H,k(H,k(i2,H)))
s2 = k(i2,k(i2,k(i2,op0))) + k(i2,k(i2,k(X,op1)))
s3 = k(i2,k(i2,k(i2,Rx(-np.pi / 4))))
s4 = k(i2,k(op0,k(i2,i2))) + k(i2,k(op1,k(i2,sX)))
s5 = k(op0,k(i2,k(i2,i2))) + k(op1,k(i2,k(i2,sX)))

slices = [s1,s2,s3,s4,s5]
s = k(i2,k(i2, k(i2,i2)))
for i in slices:
    s = np.matmul(i,s)
print(s)

state0 = k(u,k(u,k(u,u))) # |0000> or (1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
print(s.dot(state0))
print(s @ state0) # same thing fam
print(state0.T * (s @ state0)) # usual inner product 
print(np.square(np.abs(s @ state0))) # output measurement expectation vector