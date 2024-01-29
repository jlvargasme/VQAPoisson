import time

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit import transpile 

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

# VQA Poisson
from vqa_poisson import VQAforPoisson

def create_initial_state(N):
    x = np.linspace(0,1,N)
    y = np.cos(x)
    y = y / np.linalg.norm(y)
    
    return y

def experiment(bc, num_trials, num_qubits_list, num_layers, qins, optimze=False, method="bfgs"):
    
    print('-----------'+bc+' boundary condition --------------')
    
    data = {'num_qubits':[], 'obj_count':[], 'circ_count':[], 'iter_count':[], 'err':[], 'params':[], 'q_sol':[], 'cl_sol':[]}
    
    for num_qubits in num_qubits_list:
        print('-------------------------')
        print('num_qubits:', num_qubits)
        
        # set oracle for f vector
        oracle_f = QuantumCircuit(num_qubits)
        oracle_f.x(num_qubits-1)
        oracle_f.h(oracle_f.qubits)

        # change oracle to cosine function
        # y = create_initial_state(2**num_qubits)
        # init_f = QuantumCircuit(num_qubits)
        # init_f.initialize(y, range(0,num_qubits))

        # Transpile the composed circuit for execution
        # oracle_f = transpile(init_f, optimization_level=3)

    
        # set vqa instance
        vqa = VQAforPoisson(num_qubits, num_layers, bc, oracle_f=oracle_f, qinstance=qins, optimize_shift=optimze)


        obj_counts = []
        circ_counts = []
        iter_counts = []
        err = []
        params = []
        q_sol = []
    
        for seed in range(num_trials):
        
            np.random.seed(seed)
            x0 = list(4*np.pi*np.random.rand(vqa.num_params))

            # print(x0)            
            res = vqa.minimize(x0, method=method, save_logs=True)
            
    
            obj_counts.append(vqa.objective_counts)
            circ_counts.append(vqa.circuit_counts)
            iter_counts.append(len(vqa.objective_count_logs))
            err.append(vqa.get_errors(res['x'])['trace'])
            params.append(res['x'])
            q_sol.append(vqa.get_sol(res['x']).real)
            
            print('trial:', seed, 'Err.:', err[-1])
        
        data['num_qubits'].append(num_qubits)
        data['obj_count'].append(obj_counts)
        data['circ_count'].append(circ_counts)
        data['iter_count'].append(iter_counts)
        data['err'].append(err)
        data['params'].append(params)
        data['q_sol'].append(q_sol)
        data['cl_sol'].append(vqa.get_cl_sol().real)
        
    return data

# run simulation parameterized by error
pList = np.logspace(-5,1,1)
# print(pList)

# Open a file in write mode ('w')
file_b = open('baseline_error.txt', 'w')
file_o = open('optimized_error.txt', 'w')

file_b.write("writing error results for baseline simulation\n")
file_o.write("writing error results for optimized simulation\n")

# create noise model parameterized by pError
for pError in pList:
    print(f"Simulating for pError: {pError:f}")
    noise_model = NoiseModel()
    # pError = 0.0007
    error1q = depolarizing_error(pError, 1)
    error2q = depolarizing_error(pError, 2)

    noise_model.add_all_qubit_quantum_error(error1q, ['id', 'rz', 'sx', 'u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error2q, ['cx'])
    # print(noise_model)

    t0 = time.time()
    # optimizer = 'spsa'
    num_layers = 5
    num_trials = 1
    num_qubits_list = [4]
    optimize_shift=True
    # create instance with noise here
    qins = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=42, noise_model=noise_model)

    # linear shift circuit
    data_optimized = experiment('Periodic', num_trials, num_qubits_list, num_layers, qins, optimze=optimize_shift, method="powell")

    # baseline
    data_baseline = experiment('Periodic', num_trials, num_qubits_list, num_layers, qins, optimze=False, method="powell")

    # Write content to the file
    file_b.write(f"{pError:f}, {data_baseline['err'][0][0]:f}\n")
    file_o.write(f"{pError:f}, {data_optimized['err'][0][0]:f}\n")

# Close the file manually
file_b.close()
file_o.close()
