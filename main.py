from qiskit import QuantumCircuit
from helper_gates import cutWire
from helper import print_in_order, relative_error
import wire_cutting as ck
from qiskit.quantum_info import PauliList

#example use of the circuit knitting functionality


#circuit we want to estimate
qca = QuantumCircuit(9)
qca.x([0,1,2,3,4])
qca.cx(4, 5)

#circuit we want to estimate ready to be cut
qc = QuantumCircuit(10)
qc.x([0,1,2,3,4])
qc.append(cutWire, [4, 5])
qc.cx(5, 6)

print("Circuit ready to be cut-----------------------------------------------------------")
print(qc)

#get the two sub-circuits by splitting qc at the Cut marker
circs = ck.create_sub_circuits(qc)

print("Sub-circuits----------------------------------------------------------------------")
#print the sub cirsuits in the correct order
print_in_order(circs)

print("Original circuit------------------------------------------------------------------")
#make all of the circuits that we'll be sampling 
experiments = ck.make_experiments(circs)

print(qca)

print("Actual expectation values---------------------------------------------------------")
#get the actual expectation value of the circuit
actual = ck.get_actual_expvals(qca, PauliList(['IIIIIIIIZ', 'IIIIIIIZI', 'IIIIIIZII', 'IIIIIZIII', 'IIIIZIIII', 'IIIZIIIII', 'IIZIIIIII', 'IZIIIIIII', 'ZIIIIIIII']))
print(actual)


print("Estimated expectation values------------------------------------------------------")
#run the experiments and calculate the estimated expectation value
res = ck.run_experiments(experiments, 1000)
print(res)

print("Relative error--------------------------------------------------------------------")
#calculate the relative error
err = relative_error(actual, res)
print(err)