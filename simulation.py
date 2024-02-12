import time

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit import transpile 
import random

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

# VQA Poisson
from vqa_poisson import VQAforPoisson

def flatten_data(data, idx1, idx2):
#     {'num_qubits': [4],
#  'obj_count': [[0]],
#  'circ_count': [[13929]],
#  'iter_count': [[12]],
#  'err': [[0.037063796835104246]],
#  'params': [[array([ 6.28665136,  8.37627535,  7.10901932,  7.60296503,  3.90008654,
#            6.61538734,  5.32156445, 10.62748741, 12.63063255,  5.2247018 ,
#            9.40078149,  6.26734573,  7.79164364, 10.85577067,  1.69552512,
#            1.07731749,  0.52354453, 10.20921883, 10.36886821, 11.51069319,
#           12.29372414,  9.28338674,  5.00688598,  8.99931   ,  1.62426772,
#            9.42014917,  2.08121638, 12.02612882,  6.36999465,  5.52023807,
#            3.6062644 , 10.95440078,  5.33578101,  7.53382223])]],
#  'q_sol': [[array([ 0.45301963,  1.12816198,  1.67777962,  1.97077209,  1.98335126,
#            1.72506492,  1.15877648,  0.48055245, -0.4999037 , -1.17400566,
#           -1.74469377, -2.03887965, -2.07370644, -1.80841628, -1.2765805 ,
#           -0.52322544])]],
#  'cl_sol': [array([ 0.49726763,  1.24230017,  1.73857501,  1.98658842,  1.98658842,
#           1.73857501,  1.24230017,  0.49726763, -0.49726763, -1.24230017,
#          -1.73857501, -1.98658842, -1.98658842, -1.73857501, -1.24230017,
#          -0.49726763])]}
    
    num_qubits = data['num_qubits']
    obj_count = data['obj_count'][idx1][idx2]
    cir_count = data['circ_count'][idx1][idx2]
    iter_count = data['iter_count'][idx1][idx2]
    err = data['err'][idx1][idx2]
    params = data['params'][idx1][idx2].tolist()
    q_sol = data['q_sol'][idx1][idx2].tolist()
    cl_sol = data['cl_sol'][idx1][idx2].tolist()

    return {'num_qubits' : num_qubits,
             'obj_count' : obj_count,
             'cir_count' : cir_count,
             'iter_count' : iter_count,
             'err' : err,
             'params' : params,
             'q_sol' : q_sol,
             'cl_sol' : cl_sol}



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
pList = np.logspace(-5,0,15)
# print(pList)

# Open a file in write mode ('w')
file_b = open('baseline_error_qasm.txt', 'w')
file_o = open('optimized_error_qasm.txt', 'w')

file_b.write("writing error results for baseline simulation\n")
file_o.write("writing error results for optimized simulation\n")

seed_random = [42] + [random.randint(1, 100) for _ in range(5)]

for seed in seed_random:
    print(seed)
    file_b.write(f"Seed: {seed:f}\n")
    file_o.write(f"Seed: {seed:f}\n")
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
        backend = Aer.get_backend('qasm_simulator')
        num_shots=1
        # create instance with noise here
        qins = QuantumInstance(backend, seed_transpiler=seed, noise_model=noise_model, shots=num_shots)

        # linear shift circuit
        data_optimized = experiment('Periodic', num_trials, num_qubits_list, num_layers, qins, optimze=optimize_shift, method="powell")

        # baseline
        data_baseline = experiment('Periodic', num_trials, num_qubits_list, num_layers, qins, optimze=False, method="powell")
        print(f"Seed: {seed}, pError: {pError:f}, solError_b: {data_baseline['err'][0][0]:f}, solError_o: {data_optimized['err'][0][0]:f}")
        # Write content to the file
        file_b.write(f"{pError:f}, {data_baseline['err'][0][0]:f}\n")
        file_o.write(f"{pError:f}, {data_optimized['err'][0][0]:f}\n")

# Close the file manually
file_b.close()
file_o.close()


