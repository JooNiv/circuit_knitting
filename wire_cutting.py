from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit import Qubit
from itertools import product
from identity_QPD import identity_QPD2
from qiskit_aer import Aer
import numpy as np
import random
from qiskit_aer.primitives import Estimator

def get_cut_location(circuit):
    index = 0
    circuit_data = circuit.data
    cut_locations = []
    offset = 0
    while index < len(circuit):
        if circuit_data[index].operation.name == "Cut":
            qs = map(lambda x: circuit.find_bit(x).registers[0] ,circuit_data[index].qubits)
            circuit_data.remove(circuit_data[index])
            cut_locations.append((tuple(qs), index + offset))
            index -= 1
        index += 1
    return cut_locations

c = QuantumCircuit(1, name="Meas")
measure_node = c.to_instruction()

c = QuantumCircuit(1, name="Init")
initialize_node = c.to_instruction()

#Insert placeholders to the cut locations
def insert_placeholder(circuit, cut_locations):
    circuit_data = circuit.data
    offset = 0
    for i in cut_locations:
        if max(i[0][0][1], i[0][1][1]) == i[0][0][1]:
            circuit_data.insert(i[1]+offset , CircuitInstruction(operation=measure_node, qubits=[Qubit(i[0][0][0], i[0][0][1])]))
            circuit_data.insert(i[1]+offset , CircuitInstruction(operation=initialize_node, qubits=[Qubit(i[0][1][0], i[0][1][1])]))
            offset += 2
        else:
            circuit_data.insert(i[1]+offset , CircuitInstruction(operation=initialize_node, qubits=[Qubit(i[0][1][0], i[0][1][1])]))
            circuit_data.insert(i[1]+offset , CircuitInstruction(operation=measure_node, qubits=[Qubit(i[0][0][0], i[0][0][1])]))
            offset += 2
    circuit = circuit.decompose(gates_to_decompose="decomp")
    return circuit

#get number of classical bits needed
def get_num_clbits(cut_locations):
    num_clbits = 0
    max_value = max(cut_locations[0][0][0][1], cut_locations[0][0][1][1])
    for loc in cut_locations:
        if loc[0][0][1] < max_value:
            num_clbits += 1
    return num_clbits

#Cuts the given circuit into two at the location of the cut marker
def create_sub_circuits(circuit, cut_locations):
    classical_bits = get_num_clbits(cut_locations)
    sub_circuit_0 = QuantumCircuit(max(cut_locations[0][0][0][1], cut_locations[0][0][1][1]), classical_bits)
    bitindex0 = 0 #counter variable
    sub_circuit_1 = QuantumCircuit(circuit.qregs[0].size - max(cut_locations[0][0][0][1], cut_locations[0][0][1][1]), len(cut_locations) - classical_bits)
    bitindex1 = 0 #counter variable
    for i in circuit.data:
        if circuit.find_bit(i.qubits[0]).index < max(cut_locations[0][0][0][1], cut_locations[0][0][1][1]):
            test = map(lambda x: Qubit(sub_circuit_0.qregs[0], circuit.find_bit(x).index), i.qubits)
            test = tuple(test)
            classical_bits = ()
            if i.operation.name == "measure":
                classical_bits = [sub_circuit_0.clbits[bitindex0]]
                bitindex0 += 1
            sub_circuit_0.data.insert(
                len(sub_circuit_0),
                CircuitInstruction(operation=i.operation, qubits=test, clbits=classical_bits),
            )
        else:
            test = map(lambda x: Qubit(sub_circuit_1.qregs[0], circuit.find_bit(x).index - max(cut_locations[0][0][0][1], cut_locations[0][0][1][1])), i.qubits)
            test = tuple(test)
            classical_bits = ()
            if i.operation.name == "measure":
                classical_bits = [sub_circuit_1.clbits[bitindex1]]
                bitindex1 += 1
            sub_circuit_1.data.insert(
                len(sub_circuit_1),
                CircuitInstruction(operation=i.operation, qubits=test, clbits=classical_bits),
            )
    return(sub_circuit_0, sub_circuit_1)

def decomp_combinations(cut_locations):
    return list(product(identity_QPD2,repeat=len(cut_locations)))

def all_subcircs(combinations_QPD, sub_circuits, cut_locations):
    circuits = []
    coefs = []
    for j in combinations_QPD:
        sub_circuit_0 = sub_circuits[0].copy()
        sub_circuit_1 = sub_circuits[1].copy()
        count = 0
        bitind = 0
        coef = []
        for i in sub_circuit_0.data:
            if count >= len(cut_locations):
                break
            if i.operation.name == "Meas":
                ind = sub_circuit_0.data.index(i)
                sub_circuit_0.data.remove(i)
                test = map(lambda x: Qubit(sub_circuit_0.qregs[0], sub_circuit_0.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub_circuit_0.data.insert(ind, CircuitInstruction(operation=j[count]["op"], qubits=test, clbits=[sub_circuit_0.clbits[bitind]]))
                bitind += 1
                coef.append(j[count]["c"])
                count += 1
            if i.operation.name == "Init":
                ind = sub_circuit_0.data.index(i)
                sub_circuit_0.data.remove(i)
                test = map(lambda x: Qubit(sub_circuit_0.qregs[0], sub_circuit_0.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub_circuit_0.data.insert(ind, CircuitInstruction(operation=j[count]["init"], qubits=test))
                coef.append(j[count]["c"])
                count += 1
        count = 0
        bitind = 0
        for i in sub_circuit_1.data:
            if count >= len(cut_locations):
                break
            if i.operation.name == "Meas":
                ind = sub_circuit_1.data.index(i)
                sub_circuit_1.data.remove(i)
                test = map(lambda x: Qubit(sub_circuit_1.qregs[0], sub_circuit_1.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub_circuit_1.data.insert(ind, CircuitInstruction(operation=j[count]["op"], qubits=test, clbits=[sub_circuit_1.clbits[bitind]]))
                bitind += 1
                count += 1
            if i.operation.name == "Init":
                ind = sub_circuit_1.data.index(i)
                sub_circuit_1.data.remove(i)
                test = map(lambda x: Qubit(sub_circuit_1.qregs[0], sub_circuit_1.find_bit(x).index), i.qubits)
                test = tuple(test)
                sub_circuit_1.data.insert(ind, CircuitInstruction(operation=j[count]["init"], qubits=test))
                count += 1
        sub_circuit_0.measure_all()
        sub_circuit_0 = sub_circuit_0.decompose(gates_to_decompose="decomp")
        sub_circuit_1.measure_all()
        sub_circuit_1 = sub_circuit_1.decompose(gates_to_decompose="decomp")
        coefs.append(coef)
        circuits.append((sub_circuit_0, sub_circuit_1))   
    return circuits, coefs

def run_experiments(experiment_circuits):
    simulator = Aer.get_backend('aer_simulator')
    results = []
    for circuit_pair in experiment_circuits:
        result_1 = simulator.run(circuit_pair[0], shots=2**12).result().get_counts()
        result_2 = simulator.run(circuit_pair[1], shots=2**12).result().get_counts()
        results.append((result_1, result_2))
    results = modify_results(results)
    return results

def modify_results(results):
    modified_results = []
    for item in results:
        sub_results_and_probability = []
        for sub in item:
            #print(sub)
            sub_results = []
            probabilities = []
            for measurement, count in sub.items():
                separated_measurement = measurement.split() #separate the computational basis measurements at the end of circuit from the mid circuit eigen basis ones
                result = {
                    "y": [],
                    "s": []
                }
                for i in separated_measurement[0][::-1]:
                    for c in i:
                        if int(c) == 0:
                            result["y"].append(-1)
                        else:
                            result["y"].append(1)
                if(len(separated_measurement) > 1):
                    for i in separated_measurement[1][::-1]:
                        for c in i:
                            if int(c) == 0:
                                result["s"].append(-1)
                            else:
                                result["s"].append(1)
                sub_results.append(result)
                probabilities.append(count/4096)
            sub_results_and_probability.append((sub_results, probabilities))
        modified_results.append(sub_results_and_probability)
    return modified_results

def sample_results(data):
    num_qubits = len(data[0][1][0][0][0]["y"]) +len(data[0][1][1][0][0]["y"]) - 1
    num_cuts = len(data[0][1][0][0][0]["s"]) +len(data[0][1][1][0][0]["s"])
    f = np.zeros(num_qubits)
    N = min(np.power(4, (num_cuts)*2)/np.power(0.05,2)+ 100000, 1000000)
    for i in range(int(N)):
        random_coef_result_prob = random.choice(data)
        coef = random_coef_result_prob[0]
        sub_result_1 = random.choices(random_coef_result_prob[1][0][0], random_coef_result_prob[1][0][1])[0]
        sub_result_2 = random.choices(random_coef_result_prob[1][1][0], random_coef_result_prob[1][1][1])[0]
        n = []
        if (num_cuts) % 2 == 0:
            n = -1*np.concatenate((sub_result_1["y"], sub_result_2["y"][1:]))*np.prod(coef)*np.prod(sub_result_1["s"])*np.prod(sub_result_2["s"])
        else:
            n = np.concatenate((sub_result_1["y"][:-1], sub_result_2["y"]))*np.prod(coef)*np.prod(sub_result_1["s"])*np.prod(sub_result_2["s"])
        f += n
    return np.power(4, (num_cuts))*f/N

def get_actual_expvals(circ, obs):
    estimator = Estimator(run_options={"shots": None}, approximation = True)
    exact_expvals = (
        estimator.run([circ] * len(obs), list(obs)).result().values
    )
    return exact_expvals