from qiskit import QuantumCircuit

#define measurements for different bases
c = QuantumCircuit(1, name="decomp")
c.h(0)
xmeas = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.sdg(0)
c.h(0)
ymeas = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
c.initialize([1,0],0)
idmeas = c.to_instruction()

c = QuantumCircuit(1, name="decomp")
zmeas = c.to_instruction()

#define the cut location marker
cut = QuantumCircuit(2, name="Cut")
cutWire = cut.to_instruction()