from qiskit.circuit import CircuitInstruction
from qiskit.circuit import Qubit
from qiskit import QuantumCircuit
import numpy as np
from helper import to_numpy_exp
from identity_QPD import identity_QPD2
from qiskit_aer import Aer
from qiskit_aer.primitives import Estimator
from itertools import product
import random


#get the location of the cut from the circuit. Also removes the marker
def get_cut_location(circ):
    i = 0
    data = circ.data
    qss = []
    offset = 0
    while i < len(circ):
        if data[i].operation.name == "Cut":
            qs = map(lambda x: circ.find_bit(x).registers[0] ,data[i].qubits)
            data.remove(data[i])
            qss.append((tuple(qs), i + offset))
            #offset += 1
            i -= 1
        i += 1
    return qss

c = QuantumCircuit(1, name="Meas")
meas = c.to_instruction()

c = QuantumCircuit(1, name="Init")
init = c.to_instruction()

#Insert placeholders to the cut locations
def insert_placeholder(circ, qss):
    clc = QuantumCircuit(0)
    circ = circ.compose(clc)
    data = circ.data
    offset = 0
    for i in qss:
        if max(i[0][0][1], i[0][1][1]) == i[0][0][1]:
            data.insert(i[1]+offset , CircuitInstruction(operation=meas, qubits=[Qubit(i[0][0][0], i[0][0][1])]))
            data.insert(i[1]+offset , CircuitInstruction(operation=init, qubits=[Qubit(i[0][1][0], i[0][1][1])]))
            offset += 2
        else:
            data.insert(i[1]+offset , CircuitInstruction(operation=init, qubits=[Qubit(i[0][1][0], i[0][1][1])]))
            data.insert(i[1]+offset , CircuitInstruction(operation=meas, qubits=[Qubit(i[0][0][0], i[0][0][1])]))
            offset += 2
    return circ

#get number of classical bits needed
def get_num_clbits(qss):
    count = 0
    max_value = max(qss[0][0][0][1], qss[0][0][1][1])
    #print(max_value)
    for tup in qss:
        if tup[0][0][1] < max_value:
            count += 1
    return count

#Cuts the given circuit into two at the location of the cut marker
def create_sub_circuits(circ, qss):
    clb = get_num_clbits(qss)
    sub0 = QuantumCircuit(max(qss[0][0][0][1], qss[0][0][1][1]), clb)
    bitind0 = 0
    sub1 = QuantumCircuit(circ.qregs[0].size - max(qss[0][0][0][1], qss[0][0][1][1]), len(qss) - clb)
    bitind1 = 0
    for i in circ.data:
        if circ.find_bit(i.qubits[0]).index < max(qss[0][0][0][1], qss[0][0][1][1]):
            test = map(lambda x: Qubit(sub0.qregs[0], circ.find_bit(x).index), i.qubits)
            test = tuple(test)
            clb = ()
            if i.operation.name == "measure":
                clb = [sub0.clbits[bitind0]]
                bitind0 += 1
            sub0.data.insert(
                len(sub0),
                CircuitInstruction(operation=i.operation, qubits=test, clbits=clb),
            )
        else:
            test = map(lambda x: Qubit(sub1.qregs[0], circ.find_bit(x).index - max(qss[0][0][0][1], qss[0][0][1][1])), i.qubits)
            test = tuple(test)
            clb = ()
            if i.operation.name == "measure":
                clb = [sub1.clbits[bitind1]]
                bitind1 += 1
            sub1.data.insert(
                len(sub1),
                CircuitInstruction(operation=i.operation, qubits=test, clbits=clb),
            )
    return(sub0, sub1)

#make the experiments by inserting QPD into the sub-circuits
def make_experiments(circs, qss):
    comb = list(product(identity_QPD2,repeat=len(qss)))
    circuits = []
    coefs = []
    for j in comb:
        sub0 = circs[0].copy()
        sub1 = circs[1].copy()
        count = 0
        bitind = 0
        coef = []
        for i in sub0.data:
            if count >= len(qss):
                break
            if i.operation.name == "Meas":
                ind = sub0.data.index(i)
                sub0.data.remove(i)
                test = map(lambda x: Qubit(sub0.qregs[0], sub0.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub0.data.insert(ind, CircuitInstruction(operation=j[count]["op"], qubits=test, clbits=[sub0.clbits[bitind]]))
                bitind += 1
                coef.append(j[count]["c"])
                count += 1
            if i.operation.name == "Init":
                ind = sub0.data.index(i)
                sub0.data.remove(i)
                test = map(lambda x: Qubit(sub0.qregs[0], sub0.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub0.data.insert(ind, CircuitInstruction(operation=j[count]["init"], qubits=test))
                coef.append(j[count]["c"])
                count += 1
        count = 0
        bitind = 0
        for i in sub1.data:
            if count >= len(qss):
                break
            if i.operation.name == "Meas":
                ind = sub1.data.index(i)
                sub1.data.remove(i)
                test = map(lambda x: Qubit(sub1.qregs[0], sub1.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub1.data.insert(ind, CircuitInstruction(operation=j[count]["op"], qubits=test, clbits=[sub1.clbits[bitind]]))
                bitind += 1
                count += 1
            if i.operation.name == "Init":
                ind = sub1.data.index(i)
                sub1.data.remove(i)
                test = map(lambda x: Qubit(sub1.qregs[0], sub1.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub1.data.insert(ind, CircuitInstruction(operation=j[count]["init"], qubits=test))
                count += 1
        sub0.measure_all()
        sub0 = sub0.decompose(gates_to_decompose="decomp")
        sub1.measure_all()
        sub1 = sub1.decompose(gates_to_decompose="decomp")
        coefs.append(coef)
        circuits.append((sub0, sub1))   
    return circuits, coefs

#run experiments and obtain the estimated expectation values
def run_experiments(circuits, coefs, n = 1000):
    results = []
    coefs_new = []
    arr = list(zip(circuits, coefs))
    simulator = Aer.get_backend('aer_simulator')
    for i in range(8000):
        cur = random.sample(arr, 1)
        mid = []
        result = simulator.run(cur[0][0][0], shots=1).result()
        qd = list(result.get_counts().keys())[0].replace(" ", "")
        s = qd[circuits[0][0].qregs[0].size:]
        y = qd[:circuits[0][0].qregs[0].size]
        mid.append({"s": s, "y": y})
        result = simulator.run(cur[0][0][1], shots=1).result()
        qd = list(result.get_counts().keys())[0].replace(" ", "")
        s = qd[circuits[0][1].qregs[0].size:]
        y = qd[:circuits[0][1].qregs[0].size]
        mid.append({"s": s, "y": y})
        results.append(mid)
        coefs_new.append(cur[0][1])
    return results, coefs_new

def estimate_exp(results, subobs, coefs):
    f = np.zeros(len(results[0][0]["y"]) + len(results[0][1]["y"]) - 1)
    #print(f)
    for i, j in zip(results, coefs):
        #print(i)
        combined_s = np.array([list(map(lambda x: -1 if x == 0 else 1, np.array(list(i[0]["s"] + i[1]["s"]), dtype=int)))])[0]
        combined_y = []
        if subobs == 0:
            combined_y = np.array([list(map(lambda x: -1 if x == 0 else 1, np.array(list(i[0]["y"][::-1] + i[1]["y"][:-1][::-1]), dtype=int)))])[0]
            
        else:
            combined_y = np.array([list(map(lambda x: -1 if x == 0 else 1, np.array(list(i[0]["y"][1:][::-1] + i[1]["y"][::-1]), dtype=int)))])[0]
        f += np.prod(j)*np.prod(combined_s)*combined_y
            
    return f*4/8000

#get the actual expectation values of the circuit
def get_actual_expvals(circ, obs):
    estimator = Estimator(run_options={"shots": None}, approximation = True)
    exact_expvals = (
        estimator.run([circ] * len(obs), list(obs)).result().values
    )
    return exact_expvals