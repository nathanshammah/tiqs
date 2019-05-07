# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:35:53 2019

@author: fabri

This code nn produce a bloack-diagonal Hamiltonian for 
translationally invariant Liouvillians, while the quantum 
jumps will be out-of diagonal.

Include a code for generic Z_n symmetry. 
Any U(1) symmetry reduces to some Z_n once
it has been represented on a cutoff basis.
"""
from qutip import *
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time


def traslation(string):
    """ Translate string by one to the left.

    The idea is to define the symmetries using strings. therefore, the first 
    symmetry which is defined is the translational one.
    
    *Example* 
    >>> traslation('001')
    >>> '010' 

    Parameters
    ----------
    string : string
        A string representing particle states in a 1D chain.

    Returns
    -------
    traslated_string: string
        The input string shifted by one.  
    """
    traslated_string = string[1:] + string[0]
    return traslated_string



def write_state_as_number(index, N_max, lattice_size):
    """ Convert an index representation of the state into the number basis of the lattice.

    Given a Hilbert space, first enumerate them with a index. 
    The function np.base_repr(number, base) 
    returns a string representation
    of a number in the given base system. 
    This is equivalent to writing in the "number" basis our states.

    
    *Example* 
    For example, in a system of two spins, 
    the size of the basis of the Hilbert space is 4. 
    Create the following association:
    0 -> '00', 1-> '01', 2-> '10', 3 -> 11.

    >>> write_state_as_number(index=0, N_max=2, lattice_size=2)
    >>> '00' 

    Parameters
    ----------
    index : int
        The index representing a particles state in a 1D chain.

    N_max : int
        The total size of the Hilbert space of each chain site.

    lattice_size : int
        The number of sites in the 1D chain.

    Returns
    -------
    state: string
        The string representing the state in the "number (Fock) basis".  
    """    
    state = np.base_repr(index, N_max)
    for j in range(len(state), lattice_size):
        state = '0' + state
    return state

def string_to_array(string):
    """ Write the string as a list, given the string as input.

    Parameters
    ----------
    string : string
        A string representing a Fock state.
    Returns
    -------
    list: list
        The list representing the Fock state.  
    """    
    return [int(j) for j in string]

def find_representative_traslation(N_max, lattice_size):
    """ Find the representative translation.

    Use only the translational invariance. 
    Clearly, more complex symmetries can be used. 
    The Z_n symmetry can be implemented.

    Parameters
    ----------
    N_max : int
        The total size of the Hilbert space of each chain site.

    lattice_size : int
        The number of sites in the 1D chain.

    Returns
    -------
    representative: list
        The representative list.  
    """    
    # The number of element in the basis
    list_of_numbers = list(range(N_max**lattice_size))
    
    # The list to check: if the number is zero it means
    # that the state corresponding to that number is the translation of another one
    list_to_check = np.ones(N_max**lattice_size)
    
    # Create the list of representative
    representative = []

    # cycle in the number of elements
    for i in range(N_max**lattice_size):
        # check if that element is the traslation of another state
        if list_to_check[i] != 0:
            # if it is not, you eliminate all the states which can be obtained 
            # by translation 
            state = write_state_as_number(list_of_numbers[i], N_max,lattice_size)
            list_to_check[i] = 0
            representative.append(list_of_numbers[i])
            # begin translating, but check if the system is already an 
            # eigenstate              
            state_traslated = traslation(state)
            if state_traslated != state:
                number=np.sum([int(state_traslated[i])*N_max**(lattice_size - i - 1) for i in range(lattice_size)])
                list_to_check[number] = 0
                for j in range(1, lattice_size - 1):
                    state_traslated = traslation(state_traslated)
                    number = np.sum([int(state_traslated[i])*N_max**(lattice_size - i - 1) for i in range(lattice_size)])
                    list_to_check[number] = 0
    return representative

def find_representative_traslation_and_Zn(N_max, lattice_size, Z_n):
    """ Find the representative of both translation and Z_n symmetry.

    Parameters
    ----------
    N_max : int
        The total size of the Hilbert space of each chain site.

    lattice_size : int
        The number of sites in the 1D chain.

    Z_n : int
        The value of Z_n.

    Returns
    -------
    representative: list
        The representative list.  
    """    
    symmetry_dimension = Z_n
    list_of_numbers = list(range(N_max**lattice_size))
    list_to_check = np.ones(N_max**lattice_size)
    
    # Create the list of representative of each Z_n sector
    representative = []
  
    # build a symmetry sector with conserved number of particles
    for j in range(symmetry_dimension):
        representative.append([])

    # do the same as before, but this time divide in the symetry sector of Zn
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
    return representative


def rotation_matrix(N_max, lattice_size, representatives, 
    return_size=1, return_Qobj=0, Hamiltonian_size=None):
    """ Rotate the matrix to obtain it in block diagonal form.

    Having obtained the representatives, construct the rotation matrix 
    which block-diagonalise the effective Hamiltonian.
    
    Parameters
    ----------
    N_max : int
        The total size of the Hilbert space of each chain site.

    lattice_size : int
        The number of sites in the 1D chain.

    representatives : list
        The representative list.

    return_size : int
        Argument specifying if the size is to be returned.
        default : 1

    return_Qobj : int
        Argument specifying if the object is to be returned as Qobj.
        default : 0

    Hamiltonian_size
        The size of the Hamiltonian. 
        default : None  

    Returns
    -------
    rotation: ndarray matrix or :class:`qutip.Qobj`
        The rotated matrix.

    sectors: list (optional)
        The list of sectors of the block-diagonal matrix.  
    """
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
    
    # number of column of the rotation matrix
    index_of_rotation=0
               
    ## Cycle on the representatives according to the previously imposed
    # Z_n symmetry
    sectors = []
    for j in range(len(representatives)):     
        ## Go on the k space: each sector gets dived in lattice_size
        for k in klist:
            sector_dimension = 0
            ## for each state contained in the appropriate symmetyry sector,
            # construct the basis in k-space
            for state in representatives[j]:
                vector = write_state_as_number(state, N_max,lattice_size)
                vector_numpy = 0*1.j*np.ones(N_max**lattice_size)
                ## Since QuTiP uses the Fock basis as the basis to build up
                # the Hilbert space, the presence of a state translates into
                # a 1 at position 'state'                 
                vector_numpy[state] = 1 
                # translate the vector and build the appropriate basis
                for position in range(1,lattice_size):
                    vector = traslation(vector)
                    number = np.sum([int(vector[nn])*N_max**(lattice_size - nn - 1) for nn in range(lattice_size)])
                    vector_numpy[number] = vector_numpy[number] + np.exp(1.j*k*position)            
                # check that the vector is non-zero (i.e. a good vector for 
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
 
def build_appropriate_jumps(lattice_size,jump_operators, rotation):
    """ Build the appropriat, translationally-invariant jump operators. 

    Given the list of local operators and the rotation matrix,
    construct the appropriate jump operators. 
     
    Parameters
    ----------
    lattice_size : int
        The number of sites in the 1D chain.

    jump_operators : list
        The list of jump operators, each element is a :class:`qutip.Qobj` object.

    rotation: :class:`qutip.Qobj`
        The rotated matrix.  

    Returns
    -------
    appropriate_jump_operators : list 
        The list of appropriately rotated operators as :class:`qutip.Qobj` objects.

    """   
    klist = [2*np.pi*m/lattice_size for m in range(lattice_size)]
    appropriate_jump_operators = []    
    
    for k in klist:
        appropriate_jump = jump_operators[0]/np.sqrt(lattice_size)
        
        for j in range(1,lattice_size):
            appropriate_jump = appropriate_jump + np.exp(1.j*k*j)*jump_operators[j]/np.sqrt(lattice_size)    
        
        appropriate_jump = rotation*appropriate_jump*rotation.dag()
        appropriate_jump_operators.append(appropriate_jump.tidyup(atol=1e-6))
    
    return appropriate_jump_operators
    
