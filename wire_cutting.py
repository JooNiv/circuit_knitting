from qiskit.circuit import CircuitInstruction
from qiskit.circuit import Qubit
from qiskit import QuantumCircuit
import numpy as np
from helper import to_numpy_exp
from identity_QPD import identity_QPD
from qiskit_aer import Aer
from qiskit_aer.primitives import Estimator

#get the location of the cut from the circuit. Also removes the marker
def get_cut_location(circ):
    i = 0
    data = circ.data
    while i < len(circ):
        if data[i].operation.name == "Cut":
            qs = map(lambda x: circ.find_bit(x).index ,data[i].qubits)
            data.remove(data[i])
            return tuple(qs)
        i += 1
    return None

#Cuts the given circuit into two at the location of the cut marker
def create_sub_circuits(circ):
    qs = get_cut_location(circ)
    sub1 = QuantumCircuit(max(qs))
    sub2 = QuantumCircuit(circ.qregs[0].size - max(qs))
    split_circ = {
        "meas": sub2 if qs[0] == max(qs) else sub1,
        "init": sub1 if qs[0] == max(qs) else sub2,
        "index": max(qs),   
        "order": ("meas", "init") if qs[0] == min(qs) else ("init", "meas")
    }
    for i in circ:
        if circ.find_bit(i.qubits[0]).index <= qs[0]:
            test = map(lambda x: Qubit(sub1.qregs[0], circ.find_bit(x).index), i.qubits)
            test = tuple(test)
            sub1.data.insert(
                len(sub1),
                CircuitInstruction(operation=i.operation, qubits=test, clbits=sub1.clbits),
            )
        else:
            test = map(lambda x: Qubit(sub2.qregs[0], circ.find_bit(x).index - max(qs)), i.qubits)
            test = tuple(test)
            sub2.data.insert(
                len(sub2),
                CircuitInstruction(operation=i.operation, qubits=test, clbits=sub2.clbits),
            )

    return(split_circ)

#make the experiments by inserting QPD into the sub-circuits
def make_experiments(circs):
    circuits = []
    qs = (circs["meas"].qregs[0].size-1, 0) if circs["order"] == ("meas", "init") else (0, circs["init"].qregs[0].size-1)
    for i in identity_QPD:
        meas_exp = circs["meas"].copy()
        init_exp = circs["init"].copy()
        
        meas_exp.append(i["op"], [qs[0]])
        meas_exp.measure_all()
        meas_exp = meas_exp.decompose(gates_to_decompose = "decomp")
        
        mid = QuantumCircuit(init_exp.qregs[0].size)
        mid.initialize(i["rho"], [qs[1]])
        init_exp = mid.compose(init_exp)
        init_exp.measure_all()
        init_exp = init_exp.decompose(gates_to_decompose = "decomp")
        circuits.append((meas_exp, init_exp, i["c"]))
        
        
    return (circs["order"], circuits)

#run experiments and obtain the estimated expectation values
def run_experiments(circuits, n = 1000):
    f = np.zeros(circuits[1][0][0].qregs[0].size + circuits[1][0][1].qregs[0].size - 1)
    ind = 0 if circuits[0] == ("meas", "init") else circuits[1][0][0].qregs[0].size - 1
    simulator = Aer.get_backend('aer_simulator')
    for i in circuits[1]:
        for j in range(n):
            result1 = simulator.run(i[0], shots=1, memory=True).result()

            result2 = simulator.run(i[1], shots=1, memory=True).result()
            if((-1 if int(result1.get_memory(i[0])[0][ind]) == 0 else 1)*i[2] == 1):
                f += to_numpy_exp(result1.get_memory(i[0]), result2.get_memory(i[1]), circuits[0])
            else:
                f -= to_numpy_exp(result1.get_memory(i[0]), result2.get_memory(i[1]), circuits[0])
    f = f*4/(n*len(circuits[1]))
    return f

#get the actual expectation values of the circuit
def get_actual_expvals(circ, obs):
    estimator = Estimator(run_options={"shots": None}, approximation = True)
    exact_expvals = (
        estimator.run([circ] * len(obs), list(obs)).result().values
    )
    return exact_expvals