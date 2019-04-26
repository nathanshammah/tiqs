from qutip import *
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

## this is for spcific configurations, default set to false
parallel_active = False
if parallel_active == True:
    from qutip.parallel import parfor, parallel_map, serial_map

# Example: a spin chain with temperature and dissipation. 
# This chain has a U(1) symmetry.
# In the reduced basis that means a Z_n symmetry with n=lattice_size+1).

lattice_size=6
N_max=2
operators_list=[]

a = tensor(destroy(N_max), identity(N_max))
b = tensor(identity(N_max), destroy(N_max))
ide = tensor(identity(N_max), identity(N_max))

for j in range(2,lattice_size):
    a = tensor(a, identity(N_max))
    b = tensor(b, identity(N_max))
    ide = tensor(ide, identity(N_max))

operators_list.append(a)
operators_list.append(b)

for i in range(2,lattice_size):
    c = tensor(identity(N_max), identity(N_max))
    for j in range(2,lattice_size):
        if i == j:
            c = tensor(c, destroy(N_max))
        else:
            c = tensor(c, identity(N_max))
    operators_list.append(c)

omega = 1.0         # onsite energy
J = 0.5             # hopping term
gamma_p = 1.0       # incohrent pump rate
gamma_m = 1.4       # incohrent decay rate

H = 0*a
for i in range(lattice_size):
    site = operators_list[i]
    nearest = operators_list[(i + 1)%lattice_size]
    H = H + omega*(ide - 2*site.dag()*site)
    H = H + J*(site*nearest.dag() + site.dag()*nearest)
    
c_ops_minus = []
c_ops_plus = []

for j in operators_list:
    c_ops_minus.append(np.sqrt(gamma_m)*j)
    c_ops_plus.append(np.sqrt(gamma_p)*j.dag())

representatives = find_representative_traslation_and_Zn(N_max, lattice_size, lattice_size + 1)
[rotation, sectors] = rotation_matrix(N_max, lattice_size, representatives)
rotation = Qobj(rotation, dims = H.dims)
rotated_Hamiltonian = rotation*H*rotation.dag()
appropriate_jumps_minus = build_appropriate_jumps(lattice_size, c_ops_minus,rotation)
appropriate_jumps_plus = build_appropriate_jumps(lattice_size, c_ops_plus,rotation)

# now you have the "rotated_Hamiltonian" which is correctly dived in the symmetry 
# sectors, and "appropriate_jumps_minus", which describe jump between symmetry
# sectors

### visualisation
plt.matshow(np.abs(H.full()))
plt.matshow(np.abs(rotated_Hamiltonian.full()))
plt.matshow(np.abs(c_ops_minus[1].full()))
plt.matshow(np.abs(appropriate_jumps_minus[1].full()))
plt.matshow(np.abs(c_ops_plus[1].full()))
plt.matshow(np.abs(appropriate_jumps_plus[1].full()))

#check the eigenvalues graphically
plt.figure(15)
plt.plot(np.sort(rotated_Hamiltonian.eigenenergies()))
plt.plot(np.sort(H.eigenenergies()))

#and by comparing the eigenvalues
sorted_eigenvalues = np.sort(rotated_Hamiltonian.eigenenergies()), -np.sort(H.eigenenergies())
print(np.sum(np.abs(np.add(sorted_eigenvalues))))

#effect on the wavefunction
psi0 = tensor(basis(N_max,0), basis(N_max,0))

for j in range(2, lattice_size):
    psi0 = tensor(psi0, basis(N_max,0))
    
evol = -1.j*2*rotated_Hamiltonian
evol = evol.expm()    
pure_evolution = evol*psi0
pure_evolution = pure_evolution/np.sqrt(pure_evolution.norm())

plt.matshow(np.abs(pure_evolution.full()))
# effects of one jumps

for j in appropriate_jumps_plus:
    plt.matshow(np.abs(evol*j*pure_evolution.full()))

# effects of several jumps
for j in appropriate_jumps_plus:
    for k in appropriate_jumps_plus:
        plt.matshow(np.abs(evol*k*evol*j*pure_evolution.full()))
