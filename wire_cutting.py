from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit import Qubit
from itertools import product
from identity_QPD import identity_QPD2
from qiskit_aer import Aer
import numpy as np
import random
from qiskit_aer.primitives import Estimator

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

def decomp_combinations(qss):
    return list(product(identity_QPD2,repeat=len(qss)))

def all_subcircs(comb, circs, qss):
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

def run_experiments(allcircs):
    simulator = Aer.get_backend('aer_simulator')
    results = []
    for i in allcircs:
        res1 = simulator.run(i[0], shots=2**12).result().get_counts()
        res2 = simulator.run(i[1], shots=2**12).result().get_counts()
        results.append((res1, res2))
    return results

def modify_results(results):
    modified_array = []
    for item in results:
        mid = []
        for sub in item:
            #print(sub)
            rs = []
            ps = []
            for k,v in sub.items():
                kn = k.split()
                r = {
                    "y": [],
                    "s": []
                }
                for i in kn[0][::-1]:
                    for c in i:
                        if int(c) == 0:
                            r["y"].append(-1)
                        else:
                            r["y"].append(1)
                if(len(kn) > 1):
                    for i in kn[1][::-1]:
                        for c in i:
                            if int(c) == 0:
                                r["s"].append(-1)
                            else:
                                r["s"].append(1)
                rs.append(r)
                ps.append(v/4096)
            mid.append((rs, ps))
        modified_array.append(mid)
    return modified_array

def sample_results(data, qss):
    ql = len(data[0][1][0][0][0]["y"]) +len(data[0][1][1][0][0]["y"]) - 1
    f = np.zeros(ql)
    N = min(np.power(4, len(qss)*2)/np.power(0.05,2)+ 100000, 1000000)
    for i in range(int(N)):
        elem = random.choice(data)
        sub1 = random.choices(elem[1][0][0], elem[1][0][1])[0]
        sub2 = random.choices(elem[1][1][0], elem[1][1][1])[0]
        n = []
        if len(qss) % 2 == 0:
            n = -1*np.concatenate((sub1["y"], sub2["y"][1:]))*np.prod(elem[0])*np.prod(sub1["s"])*np.prod(sub2["s"])
        else:
            n = np.concatenate((sub1["y"][:-1], sub2["y"]))*np.prod(elem[0])*np.prod(sub1["s"])*np.prod(sub2["s"])
        f += n
    return np.power(4, len(qss))*f/N

def get_actual_expvals(circ, obs):
    estimator = Estimator(run_options={"shots": None}, approximation = True)
    exact_expvals = (
        estimator.run([circ] * len(obs), list(obs)).result().values
    )
    return exact_expvals