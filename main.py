from qiskit import QuantumCircuit
from helper_gates import cutWire
from helper import relative_error
import wire_cutting as ck
from qiskit.quantum_info import PauliList

#example use of the circuit knitting functionality

#original full circuit
qc = QuantumCircuit(3)
qc.x(0)
qc.cx(0,1)
qc.cx(1,2)
qc.cx(0,1)

print("Original circuit-------------------------------------------------------------------------------------")
print(qc)

#circuit ready for partitioning with cutWire instructions inserted
qca = QuantumCircuit(4)
qca.x(0)
qca.cx(0,1)
qca.append(cutWire, [1,2])
qca.cx(2,3)
qca.append(cutWire, [2,1])
qca.cx(0,1)

print("Circuit with cut instructions-------------------------------------------------------------------------")
print(qca)

#get locations of the cuts
qss = ck.get_cut_location(qca)

#print(qss)

#insert placeholders for the masure and initialize nodes
circ = ck.insert_placeholder(qca, qss)
circ = circ.decompose(gates_to_decompose="decomp")

print("Circuit ready to be cut-------------------------------------------------------------------------------")
print(circ)

#make the subcircuits
circs = ck.create_sub_circuits(circ, qss)

print("Sub-circuits------------------------------------------------------------------------------------------")
print(circs[0])
print(circs[1])

#get all the combinations from the QPD
ds = ck.decomp_combinations(qss)

#make the experiment circuits
allcircs, coefs = ck.all_subcircs(ds, circs, qss)

#run the experiments
results = ck.run_experiments(allcircs)

#apply the {0,1} -> [-1,1] post-processing function
modres = ck.modify_results(results)

#zip coefficients and results together
data = list(zip(coefs, modres))

#sample the experiment results and calculate the expectation value
f = ck.sample_results(data, qss)
print("Estimated expectation values--------------------------------------------------------------------------")
print(f)

#calculate the actual expectation values
actual = ck.get_actual_expvals(qc, PauliList(["IIZ", "IZI", "IIZ"]))
print("Actual expectation values-----------------------------------------------------------------------------")
print(actual)

#calculate the relative error
err = relative_error(actual, f)
print("Relative error----------------------------------------------------------------------------------------")
print(err)
print("Absolute difference-----------------------------------------------------------------------------------")
print(abs(actual-f))