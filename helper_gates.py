from qiskit import QuantumCircuit
import numpy as np

#define measurements for different bases
c = QuantumCircuit(1,1, name="decomp")
c.h(0)
c.measure(0,0)
xmeas = c.to_instruction()

c = QuantumCircuit(1,1, name="decomp")
c.sdg(0)
c.h(0)
c.measure(0,0)
ymeas = c.to_instruction()

c = QuantumCircuit(1,1, name="decomp")
c.initialize([1,0],0)
c.measure(0,0)
idmeas = c.to_instruction()

c = QuantumCircuit(1,1, name="decomp")
c.measure(0,0)
zmeas = c.to_instruction()

#define the cut location marker
cut = QuantumCircuit(2, name="Cut")
cutWire = cut.to_instruction()

#define initialization instructions for the eigenstates
c = QuantumCircuit(1, name="decomp")
c.initialize([1,0], 0)
zero_init = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.initialize([0,1], 0)
one_init = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.initialize([1,1]*1/np.sqrt(2), 0)
plus_init = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.initialize([1,-1]*1/np.sqrt(2), 0)
minus_init = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.initialize([1,1j*1]*1/np.sqrt(2), 0)
i_plus_init = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.initialize([1,-1j*1]*1/np.sqrt(2), 0)
i_minus_init = c.to_instruction()
