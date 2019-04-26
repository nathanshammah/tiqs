# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:35:53 2019

@author: fabri

This code will produce a bloack-diagonal Hamiltonian for 
translationally invariant Liouvillians, while the quantum 
jumps will be out-of diagonal.

We also consider a code for generic Z_n symmetry, where 
we stress that any U(1) symmetry reduces to some Z_n once
it has been represented on a cutoff basis.
"""
from qutip import *
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

# The idea is to define the symmetries using strings. therefore, the first 
# simmetry which we define is the translational one
## example: given the string '001', it will return '010'
def traslation(string):
    traslated_string = string[1:] + string[0]
    return traslated_string
####    
# Here we define the function write_state_as_number. Given a Hilbert space, 
# we enumerate them with a index. The function np.base_repr(number, base) 
# return a string representation of a number in the given base system.
# This is equivalent to writing in the "number" basis our states.

# For example, in a system of two spin, the size of the basis of the Hilbert 
# space is 4. We will create the following association:
# 0 -> '00', 1-> '01', 2-> '10', 3 -> 11.
def write_state_as_number(index, N_max, lattice_size):
    state = np.base_repr(index, N_max)
    for j in range(len(state),lattice_size):
        state = '0' + state
    return state

# This function instead writes the string as an array, given the string as input.     
def string_to_array(string):
    return [int(j) for j in string]

## Here we use only the translational invariance. Clearely, more complex simmetries
# can be used (see e.g. the code below, where the Z_n) symmetry is implemented)
def find_representative_traslation(N_max, lattice_size):
    #this is the number of element in the basis
    list_of_numbers = list(range(N_max**lattice_size))
    
    #this is the list which needs to be checked: if the number is zero it means
    #that the state corresponding to that number is the translation of another one
    list_to_check = np.ones(N_max**lattice_size)
    
    #create the list of representative
    representative = []

    #cycle in the number of elements
    for i in range(N_max**lattice_size):
        #check if that element is the traslation of another state
        if list_to_check[i] != 0:
            #if it is not, you eliminate all the state which can be obtained 
            #by translation 
            state = write_state_as_number(list_of_numbers[i], N_max,lattice_size)
            list_to_check[i] = 0
            representative.append(list_of_numbers[i])
            # we start translating, but we check if the system is already an 
            # eigenstate              
            state_traslated = traslation(state)
            if state_traslated != state:
                number=np.sum([int(state_traslated[i])*N_max**(lattice_size - i - 1) for i in range(lattice_size)])
                list_to_check[number] = 0
                for j in range(1, lattice_size - 1):
                    state_traslated = traslation(state_traslated)
                    number = np.sum([int(state_traslated[i])*N_max**(lattice_size - i - 1) for i in range(lattice_size)])
                    list_to_check[number] = 0
    return(representative)

##here we use both the translational invariance and a Z_n symmetry
def find_representative_traslation_and_Zn(N_max, lattice_size, Z_n):
    symmetry_dimension = Z_n
    list_of_numbers = list(range(N_max**lattice_size))
    list_to_check = np.ones(N_max**lattice_size)
    
    #create the list of representative of each Z_n sector
    representative = []
  
    #build a symmetry sector with conserved number of particles
    for j in range(symmetry_dimension):
        representative.append([])

    #do the same as before, but this time divide in the symetry sector of Zn
    for i in range(N_max**lattice_size):
        if list_to_check[i] !=0:
            state = write_state_as_number(list_of_numbers[i], N_max,lattice_size)
            list_to_check[i] = 0
            sector_value = np.sum([int(state[i]) for i in range(lattice_size)])%symmetry_dimension
            representative[sector_value].append(list_of_numbers[i]) 
            state_traslated = traslation(state)
            
            if state_traslated != state:
                number = np.sum([int(state_traslated[i])*N_max**(lattice_size - i - 1) for i in range(lattice_size)])
                list_to_check[number] = 0
                for j in range(1, lattice_size - 1):
                    state_traslated = traslation(state_traslated)
                    number = np.sum([int(state_traslated[i])*N_max**(lattice_size - i - 1) for i in range(lattice_size)])
                    list_to_check[number] = 0
    return(representative)

# having obtained the representatives we can construct the rotation matrix wich 
# block diagonalise the effective Hamiltonian
def rotation_matrix(N_max, lattice_size, representatives, 
    return_size=1, return_Qobj=0, Hamiltonian_size=None):

    if Hamiltonian_size == None:
        Hamiltonian_size = N_max**lattice_size

    # initialise the matrix
    rotation = 0*1.j*np.ones([N_max**lattice_size, N_max**lattice_size])

    # build the structure of the basis element
    basis_structure = tensor(basis(N_max,0), basis(N_max,0))
    for j in range(2,lattice_size):
        basis_structure = tensor(basis_structure, basis(N_max,0))
    basis_structure = 0*basis_structure
    
    # list of the kappa for the translations
    klist = [2*np.pi*m/lattice_size for m in range(lattice_size)]
    
    #number of column of the rotation matrix
    index_of_rotation=0
               
    ## we cycle on the representatives according to the previously imposed
    # Z_n symmetry
    sectors = []
    for j in range(len(representatives)):     
        ## we go on the k space: each sector gets dived in lattice_size
        for k in klist:
            sector_dimension = 0
            ## for each state contained in the appropriate symmetyry sector,
            # we construct the basis in k-space
            for state in representatives[j]:
                vector = write_state_as_number(state, N_max,lattice_size)
                vector_numpy = 0*1.j*np.ones(N_max**lattice_size)
                ## Since qutip uses the Fock basis as the basis to build up
                # the Hilbert space, the presence of a state translates into
                # a 1 at position 'state'                 
                vector_numpy[state] = 1 
                # we now translate the vector and build the appropriate basis
                for position in range(1,lattice_size):
                    vector = traslation(vector)
                    number = np.sum([int(vector[nn])*N_max**(lattice_size - nn - 1) for nn in range(lattice_size)])
                    vector_numpy[number] = vector_numpy[number] + np.exp(1.j*k*position)            
                #we check that the vector is non-zero (i.e. a good vector for 
                # the translation sector)
                if np.sum(np.abs(vector_numpy)) >= 0.5:
                    rotation[index_of_rotation] = np.round(vector_numpy/np.sqrt(np.dot(np.conj(vector_numpy),vector_numpy)), 12)
                    index_of_rotation = index_of_rotation + 1
                    sector_dimension = sector_dimension + 1
            sectors.append(sector_dimension)

    if return_Qobj:
        rotation = Qobj(rotation, dims = Hamiltonian_size)
    if return_size:
        return rotation, sectors
    else:
        return rotation

# given the list of local operators, we build up the operators which are 
# translationally invariant  
### RMK: rotation must be a Qobj from qutip!!!
def build_appropriate_jumps(lattice_size,jump_operators, rotation):   
    klist = [2*np.pi*m/lattice_size for m in range(lattice_size)]
    appropriate_jump_operators = []    
    for k in klist:
        appropriate_jump = jump_operators[0]/np.sqrt(lattice_size)
        
        for j in range(1,lattice_size):
            appropriate_jump = appropriate_jump + np.exp(1.j*k*j)*jump_operators[j]/np.sqrt(lattice_size)    
        appropriate_jump = rotation*appropriate_jump*rotation.dag()
        appropriate_jump_operators.append(appropriate_jump.tidyup(atol=1e-6))
    return appropriate_jump_operators
    