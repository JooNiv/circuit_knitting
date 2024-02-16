from qiskit import QuantumCircuit
from helper_gates import cutWire
from helper import relative_error
import wire_cutting as ck
from qiskit.quantum_info import PauliList
import numpy as np

#example use of the circuit knitting functionality


#circuit we want to estimate
qca = QuantumCircuit(3)
qca.rx(np.pi/4, 0)
qca.rx(np.pi/4, 1)
qca.cx(0,1)
qca.cx(1,2)
qca.rx(np.pi/4, 2)

#circuit we want to estimate ready to be cut
qc = QuantumCircuit(4)
qc.rx(np.pi/4, 0)
qc.rx(np.pi/4, 1)
qc.cx(0,1)
qc.append(cutWire, [1,2])
qc.cx(2,3)
qc.rx(np.pi/4, 3)

print("Circuit ready to be cut-----------------------------------------------------------")
print(qc)

print("Get cut locations-----------------------------------------------------------------")
qss = ck.get_cut_location(qc)

print("Insert placeholders---------------------------------------------------------------")
circ = ck.insert_placeholder(qc, qss)
circ = circ.decompose(gates_to_decompose="decomp")
print(circ)

#get the two sub-circuits by splitting qc at the Cut marker
circs = ck.create_sub_circuits(circ, qss)

print("Sub-circuits----------------------------------------------------------------------")
#print the sub cirsuits in the correct order
print(circs[0])
print(circs[1])

print("Original circuit------------------------------------------------------------------")
#make all of the circuits that we'll be sampling 
experiments, coefs = ck.make_experiments(circs, qss)

print(qca)

print("Actual expectation values---------------------------------------------------------")
#get the actual expectation value of the circuit
actual = ck.get_actual_expvals(qca, PauliList(['IIZ', 'IZI', 'ZII']))
print(actual)


print("Estimated expectation values------------------------------------------------------")
#run the experiments and calculate the estimated expectation value
res, coefs = ck.run_experiments(experiments, coefs)
est = ck.estimate_exp(res, len(qss)%2, coefs)
print(est)

print("Relative error--------------------------------------------------------------------")
#calculate the relative error
err = relative_error(actual, est)
print(err)