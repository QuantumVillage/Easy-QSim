# 
#       #### QUANTUM VILLAGE ####
# 
# ________/\\\________/\\\________/\\\_        
#  _____/\\\\/\\\\____\/\\\_______\/\\\_       
#   ___/\\\//\////\\\__\//\\\______/\\\__      
#    __/\\\______\//\\\__\//\\\____/\\\___     
#     _\//\\\______/\\\____\//\\\__/\\\____    
#      __\///\\\\/\\\\/______\//\\\/\\\_____   
#       ____\////\\\//_________\//\\\\\______  
#        _______\///\\\\\\_______\//\\\_______ 
#         _________\//////_________\///________
# 
# Easy-Sim v0.1 (Aug 2022)
# 
# A minimal but 'properly accurate' quantum simulator
# written in python and wordy comments! 
# 
# Author: Mark C.
# 
# Description: By 'properly accurate' we mean that complex
# values are used and 2x2 unitary gate matrices are composed
# in order to determine the final statevector for a 
# given quantum circuit - in this case the CHSH.


# Usual imports for data and maths-y stuff...
import numpy as np
from numpy import ndarray

# This is just to make some command line stuff look pretty...
np.set_printoptions(edgeitems=30, linewidth=100000)

# so we make a new data view for Numpy - it extends ndarray (the default)
# by adding a conjugate transpose - something that our unitary gates will need! 
class simarr(ndarray):
    @property
    def H(self):
        return self.conj().T

# So first we define the |0> and |1> kets as 'up' and 'down'
# because, intuitively, that's where they point on the Bloch sphere.
u = np.array([[1+0j],[0+0j]]).view(simarr) # up qbit 
d = np.array([[0+0j],[1+0j]]).view(simarr) # down qbit
# This is just the 2 by 2 identity matrix that we will use for 'blank'
# in our circuits.
i2 = np.eye(2)

# We will use the Kroenecker product A LOT so make a pointer to the function
# that is just a letter. 
k = np.kron
# We will need this outer products (of |0><0| and |1><1| respectively) to 
# create controlled gates in our circuits.
op0 = k(u, u.H)
op1 = k(d, d.H)

# Now for some gates! First up, the quantum NOT gate:
X = np.array([[0,1],[1,0]]).view(simarr)
# And, because the quantum NOT gate is a matrix, we can have
# the sqrt(NOT) gate... so let's get that:
sX = np.array([[(1+1j)/2,(1-1j)/2],
              [(1-1j)/2,(1+1j)/2]]).view(simarr)
# Next the Hadamard gate - this puts a qubit into superposition, 
# and applying it again takes the qubit out of superposition.
H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)]]).view(simarr)
# Now a rotation on X axis gates. This takes an argument so let's prepare that:
def Rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]]).view(simarr)
# Similar story for a parameterized Y-rotation gate:
def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]]).view(simarr)

# So here is the circuit we want to simulate. Notice that 
# a circuit can be seen as a sequence of 'slices' (called other names in
# other quantum sims) that you take the starting statevector, multiply it by
# the matrix of the first slice, then take that output and multiply it by
# the matrix of the next slice and continue to the end. Because of 
# ~*_==maths==_*~ this will work the same as 'multitply all the matrices together
# then multiply the initial state by that matrix'. 

"""
Circuit Diagram - CHSH:
 s1 s2 s3 s4 s5 s6 s7
-H--c--Rx-sX-sX-
----X-----|--|--
-H--------c--|--
-H-----------c--
"""
# To get each successive matrix, we just take the Kroenecker product of
# the matrix of each, noting that the controlled gates have two matrices 
# (one for the control being off and one for the control being on). 
# NB - the order is 'bottom to top' for the left-to-right matrix order.
s1 = k(H,k(H,k(i2,H)))
s2 = k(i2,k(i2,k(i2,op0))) + k(i2,k(i2,k(X,op1)))
s3 = k(i2,k(i2,k(i2,Rx(-np.pi / 4))))
s4 = k(i2,k(op0,k(i2,i2))) + k(i2,k(op1,k(i2,sX)))
s5 = k(op0,k(i2,k(i2,i2))) + k(op1,k(i2,k(i2,sX)))

# So now we have all of our slices, we take an initial identity matrix
# ('s' in the code below) and use that as our starting point. 
slices = [s1,s2,s3,s4,s5]
s = k(i2,k(i2, k(i2,i2)))
for i in slices:
    s = np.matmul(i,s)
# by this point we have the state matrix for our circuit! 
print(s)

# Now we set the initial statevector as 'all zeroes' or '(1,0,0,0...,0)'
state0 = k(u,k(u,k(u,u))) # |0000> or (1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
print(s.dot(state0))
# Here's a fun little bit of python notation:
print(s @ state0) # same thing fam
print(state0.T * (s @ state0)) # usual inner product 
# Now take the values of the statevector and square the absolute value (of complex numbers) 
# AND WE'RE DONE!!
print(np.square(np.abs(s @ state0))) # output measurement expectation vector
