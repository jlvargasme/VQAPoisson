import warnings

import numpy as np
from scipy.optimize import minimize
from qiskit.algorithms.optimizers import SPSA
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.utils import QuantumInstance
import qiskit.quantum_info as qi


class VQAforPoisson():
    """ Variational quantum algorithm for Poisson equation based on the minimum potential energy
    
    :class:`VQAforPoisson` object works with class: `qiskit.aqua.QuantumInstance` or `backend`.
    
    >>> from vqa_poisson import VQAforPoisson
    >>> num_qubits = ... # int
    >>> num_layers = ... # int
    >>> bc = ... # str
    >>> oracle_f = ... # qiskit.QuantumCircuit
    >>> qins = ... # qiskit.aqua.QuantumInstance
    >>> vqa = VQAforPoisson(num_qubits, num_layers, bc, oracle_f=oracle_f, qinstance=qins)
    >>> x0 = ... # numpy.ndarray
    >>> res = vqa.minimize(x0)
    """

    def __init__(self, num_qubits, num_layers, bc, *, backend=None, qinstance=None, oracle_f=None, c=1e-3, use_mct_ancilla=False, optimize_shift=False):

        """
        Parameters
        ----------
        num_qubits
            The number of qubits :int.
        num_layers
            The number of layers of parameterized quantum circuit :int.
        bc
            A type of boundary conditions 'Periodic', 'Dirichlet', or 'Neumann'.
        backend
            A backend to be used. This parameter will be active if qinstance is None.
        qinstance
            A quantum instance object :qiskit.aqua.QuantumInstance.
        oracle_f
            A quantum circuit for encoding a source for the Poisson equation :qiskit.QuantumCircuit.
        c
            A parameter to avoid singularity of the stiffness matrix when bc = 'Periodic' or 'Neumann' :float.
        use_mct_ancilla
            A flag to switch whether an ancilla qubit is used to implement a multi-controlled Toffoli gate :bool:
        optimize_shift
            A flag to switch between O(n^2) and O(n) implementation of the shift operator :bool:
        """
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.bc = bc
        self.c = c
        self.use_mct_ancilla = use_mct_ancilla
        self.optimize_shift = optimize_shift

        if self.num_qubits <= 3 and self.use_mct_ancilla:
            warnings.warn("mct with ancilla qubits is valid for num_qubits > 3, so 'use_mct_ancilla' has changed to 'False'.")
            self.use_mct_ancilla=False

        if use_mct_ancilla:
            self.num_mct_ancilla = self.num_qubits - 3
        else:
            self.num_mct_ancilla = 0

        self.num_params_per_layer = 2*(self.num_qubits//2) + 2*((self.num_qubits-1)//2)
        self.num_params = num_qubits+num_layers*self.num_params_per_layer
        self.circuit_counts = 0

        self.qreg = QuantumRegister(num_qubits, 'q')
        self.qreg_ancilla = QuantumRegister(1, 'q_ancilla')
        self.inc_ancilla = QuantumRegister(num_qubits - 2, 'inc_ancilla') # ancilla for linear shift operator

        if oracle_f is not None:
            self.qc_f_vec = oracle_f
        else:
            self.qc_f_vec = QuantumCircuit(self.qreg)

        if qinstance is None:
            self.qinstance = QuantumInstance(backend, seed_simulator=42, seed_transpiler=42, shots=8192)
        else:
            self.qinstance = qinstance

    def objective(self, params):

        obj = self.evaluate(params)[0]
        self.current_objective = obj
        # print(obj)
        return obj

    def evaluate(self, params):

        A0_X = self._calc_Xn(params, is_shift=False)
        A1_X = self._calc_Xn(params, is_shift=True)

        if self.bc == 'Periodic':
            B = 0
            c = self.c
        elif self.bc == 'Dirichlet':
            B_X = self._calc_for_bc(params, is_identity=False)
            B = -B_X
            c = self.c
        elif self.bc == 'Neumann':
            B_I = self._calc_for_bc(params, is_identity=True)
            B_X = self._calc_for_bc(params, is_identity=False)
            B = B_I - B_X
            c = self.c
        else:
            raise ValueError('Invalid boundary condition setting. Boundary condition is either "Periodic", "Neumann", or "Dirichlet".')
        
        # this computes the expectation of A, see Ipad notes for derivation details
        # in summary, this used the formulas for expectation of identity and shifted identities, along with linearity of operators
        A = 2 - A0_X - A1_X - B + c

        X_In = self._calc_X0(params)
        r = X_In / A
        obj = -0.5*X_In**2 / A

        return obj, r, A, X_In

    def ansatz(self, qc, params, *, control=None):

        params = [params[:self.num_qubits]] \
                + [params[self.num_qubits+i_layer*self.num_params_per_layer:self.num_qubits+(i_layer+1)*self.num_params_per_layer] \
                    for i_layer in range(self.num_layers)]

        if control is None:
            for i in range (self.num_qubits):
                qc.ry(params[0][i], self.qreg[i])
            for i_layer in range(self.num_layers):
                for i in range(self.num_qubits//2):
                    qc.cz(self.qreg[2*i], self.qreg[2*i+1])
                    qc.ry(params[i_layer+1][2*i], self.qreg[2*i])
                    qc.ry(params[i_layer+1][2*i+1], self.qreg[2*i+1])
                for i in range((self.num_qubits-1)//2):
                    qc.cz(self.qreg[2*i+1], self.qreg[2*i+2])
                    qc.ry(params[i_layer+1][2*(self.num_qubits//2)+2*i], self.qreg[2*i+1])
                    qc.ry(params[i_layer+1][2*(self.num_qubits//2)+2*i+1], self.qreg[2*i+2])
        else:
            for i in range (self.num_qubits):
                qc.cry(params[0][i], control, self.qreg[i])
            for i_layer in range(self.num_layers):
                for i in range(self.num_qubits//2):
                    qc.mcp(np.pi, control+[self.qreg[2*i]], self.qreg[2*i+1])
                    qc.cry(params[i_layer+1][2*i], control, self.qreg[2*i])
                    qc.cry(params[i_layer+1][2*i+1], control, self.qreg[2*i+1])
                for i in range((self.num_qubits-1)//2):
                    qc.mcp(np.pi, control+[self.qreg[2*i+1]], self.qreg[(2*i+2)%self.num_qubits])
                    qc.cry(params[i_layer+1][2*(self.num_qubits//2)+2*i], control, self.qreg[2*i+1])
                    qc.cry(params[i_layer+1][2*(self.num_qubits//2)+2*i+1], control, self.qreg[2*i+2])

        return qc

    def state_preparation(self, qc, *, zero_state='f_vec', one_state='ansatz', params=None, dparams=None):

        assert zero_state in ['ansatz', 'grad_ansatz', 'f_vec'], " 'zero_state' must be either 'ansatz', 'grad_ansatz', or 'f_vec'. "
        assert one_state in ['ansatz', 'grad_ansatz', 'f_vec'], " 'one_state' must be either 'ansatz', 'grad_ansatz', or 'f_vec'. "

        if 'ansatz' in [zero_state, one_state]:
            assert params is not None, " 'params' is required for 'ansatz'. "
        if 'grad_ansatz' in [zero_state, one_state]:
            assert dparams is not None, " 'dparams' is required for 'grad_ansatz'. "

        qc.h(self.qreg_ancilla)
        if zero_state == 'ansatz':
            qc = self.ansatz(qc, params, control = self.qreg_ancilla[::])
        elif zero_state == 'grad_ansatz':
            qc = self.ansatz(qc, dparams, control = self.qreg_ancilla[::])
        elif zero_state == 'f_vec':
            qc.compose(self.qc_f_vec.control(1), self.qreg_ancilla[::]+self.qreg[::], inplace=True)

        qc.x(self.qreg_ancilla)
        if one_state == 'ansatz':
            qc = self.ansatz(qc, params, control=self.qreg_ancilla[::])
        elif one_state == 'grad_ansatz':
            qc = self.ansatz(qc, dparams, control=self.qreg_ancilla[::])
        elif one_state == 'f_vec':
            qc.compose(self.qc_f_vec.control(1), self.qreg_ancilla[::]+self.qreg[::], inplace=True)

        return qc

    def create_incrementer(self, N):
    # experiment from O(n) incrementer circuit
    # uses n - 2 ancilla qubits
        
        if N == 1:
            qc = QuantumCircuit(1)
            qc.x(0)
            return qc
        elif N == 2:
            qc = QuantumCircuit(2)
            qc.cx(0,1)
            qc.x(0)
            return qc
        
        nq_total = N + (N - 2)

        qc = QuantumCircuit(nq_total)

        # add Toffoli gates
        for i in range(0,N-2):
            qc.ccx(i*2,i*2 + 1, i*2 + 2)
        qc.cx(nq_total - 2, nq_total - 1)
        # qc.barrier()

        # add CX gates
        for i in range(N-3,-1,-1):
            qc.ccx(i*2,i*2 + 1, i*2 + 2)
            qc.cx(i*2, i*2 + 1)

        qc.x(0)

        return qc
    
    def qubit_list(self):

        q_list = [self.qreg[0]]
        for i in range(len(self.qreg)-2):
            q_list.append(self.qreg[i+1])
            q_list.append(self.inc_ancilla[i])
        q_list.append(self.qreg[-1])

        return q_list

    def shift_add(self, qc):
        
        if self.optimize_shift:
            # implement linear shift operator
            qc.compose(self.create_incrementer(self.num_qubits), qubits=self.qubit_list(), inplace=True)

        else:
            if not self.use_mct_ancilla:
                for i in reversed(range(1, self.num_qubits)):
                    qc.mct(self.qreg[:i], self.qreg[i])
                qc.x(self.qreg[0])
            else:
                qreg_shift_ancilla = QuantumRegister(self.num_qubits-3, 'q_shift_ancilla')
                qc.add_register(qreg_shift_ancilla)
                for i in reversed(range(1, self.num_qubits)):
                    qc.mct(self.qreg[:i], self.qreg[i], qreg_shift_ancilla, mode='v-chain')
                qc.x(self.qreg[0])

        return qc

    def grad(self, params):

        _, _, A, X_In = self.evaluate(params)

        dobj = []
        for idx in range(len(params)):
            dparams = params.copy()
            dparams[idx] += np.pi

            dA0_X = self._calc_grad_A(params, dparams, is_shift=False)
            dA1_X = self._calc_grad_A(params, dparams, is_shift=True)

            if self.bc == 'Periodic':
                dB = 0
            elif self.bc == 'Neumann':
                dB_I = self._calc_grad_for_bc(params, dparams, is_identity=True)
                dB_X = self._calc_grad_for_bc(params, dparams, is_identity=False)
                dB = dB_I - dB_X
            elif self.bc == 'Dirichlet':
                dB_X = self._calc_grad_for_bc(params, dparams, is_identity=False)
                dB = -dB_X
            else:
                raise ValueError('Invalid boundary condition setting. Boundary condition is either "Periodic", "Neumann", or "Dirichlet".')

            dA = 0.5*(- dA0_X - dA1_X - dB)
            dX_In = 0.5*self._calc_X0(dparams)
            dobj += [-X_In*dX_In/A + (X_In**2)*dA/A**2]

        return dobj

    def _calc_Xn(self, params, *, is_shift=False):

        if self.optimize_shift:
            qc = QuantumCircuit(self.qreg, self.inc_ancilla)
        else:
            qc = QuantumCircuit(self.qreg)

        qc = self.ansatz(qc, params)
        # print(qc)

        if is_shift:
            qc = self.shift_add(qc)
            # print(qc)

        qc.h(self.qreg[0])
        if self.qinstance.is_statevector:
        # simulator
            sv = self.qinstance.execute(qc).get_statevector(qc)
            val = 0
            len_sv = len(sv)
            qc_num = (int) (qc.num_qubits)

            if self.optimize_shift: # last n - 2 qubits do not contribute to overall result of simulation
                qc_num = (int) (qc_num/2 + 1) # get number of qubits from n + (n - 2) = qc_num
                inc_ancilla = (int) (qc_num - 2)
                len_sv = (int) (len_sv >> inc_ancilla)

            for l in range(len_sv):
                bits = bin(l)[2:].zfill(qc_num)
                if bits[-1] == '0':
                    val += np.real(sv[l]*sv[l].conjugate())
                elif bits[-1] == '1':
                    val -= np.real(sv[l]*sv[l].conjugate())
        else:
        # hardware
            creg = ClassicalRegister(1, 'c')
            qc.add_register(creg)
            qc.measure(self.qreg[0], creg)
            counts = self.qinstance.execute(qc).get_counts()

            for key in ['0', '1']:
                if key not in counts.keys():
                    counts[key] = 0
            val = (counts['0'] - counts['1']) / float(self.qinstance.run_config.shots)

        self.circuit_counts += 1

        return val

    def _calc_for_bc(self, params, *, is_identity=False):

        if self.optimize_shift:
            qc = QuantumCircuit(self.qreg, self.inc_ancilla)
        else:
            qc = QuantumCircuit(self.qreg)

        qc = self.ansatz(qc, params)
        qc = self.shift_add(qc)

        if not is_identity:
            qc.h(self.qreg[0])

        if self.qinstance.is_statevector:
            sv = self.qinstance.execute(qc).get_statevector(qc)
            val = 0

            len_sv = len(sv)
            qc_num = (int) (qc.num_qubits)

            if self.optimize_shift: # last n - 2 qubits do not contribute to overall result of simulation
                qc_num = (int) (qc_num/2 + 1) # get number of qubits from n + (n - 2) = qc_num
                inc_ancilla = (int) (qc_num - 2)
                len_sv = (int) (len_sv >> inc_ancilla)

            for l in range(len_sv):
                bits = bin(l)[2:].zfill(qc_num)
                if not np.array([int(bits[i]) for i in range(len(bits)-1)]).any():
                    if is_identity:
                        val += np.real(sv[l]*sv[l].conjugate())
                    else:
                        if bits[-1] == '0':
                            val += np.real(sv[l]*sv[l].conjugate())
                        else:
                            val -= np.real(sv[l]*sv[l].conjugate())
        else:
            creg = ClassicalRegister(self.num_qubits, 'c')
            qc.add_register(creg)
            qc.measure(self.qreg, creg)
            counts = self.qinstance.execute(qc).get_counts()
            for key in ['0'*(self.num_qubits-1)+'0', '0'*(self.num_qubits-1)+'1']:
                if key not in counts.keys():
                    counts[key] = 0
            if is_identity:
                val = (counts['0'*(self.num_qubits-1)+'0'] + counts['0'*(self.num_qubits-1)+'1']) / float(self.qinstance.run_config.shots)
            else:
                val = (counts['0'*(self.num_qubits-1)+'0'] - counts['0'*(self.num_qubits-1)+'1']) / float(self.qinstance.run_config.shots)

        self.circuit_counts += 1

        return val

    def _calc_X0(self, params):
    # This function computes the expectation value for
    # 
    # X_In = < f, Psi(\theta) | X (x) I^(x)n | f, Psi(\theta) >
    #

        qc = QuantumCircuit(self.qreg, self.qreg_ancilla)
        # create Hadamard test circuit to compute expectation value
        qc = self.state_preparation(qc, zero_state='f_vec', one_state='ansatz', params=params)

        qc.h(self.qreg_ancilla[0])
        if self.qinstance.is_statevector:
        # compute expectation value in simulator
            sv = self.qinstance.execute(qc).get_statevector(qc)
            val = 0
            for l in range(len(sv)):
                bits = bin(l)[2:].zfill(qc.num_qubits)
                if bits[0] == '0':
                    val += np.real(sv[l]*sv[l].conjugate())
                elif bits[0] == '1':
                    val -= np.real(sv[l]*sv[l].conjugate())
        else:
        # compute expectation value in hardware
            creg = ClassicalRegister(1, 'c')
            qc.add_register(creg)
            qc.measure(self.qreg_ancilla, creg[0])
            counts = self.qinstance.execute(qc).get_counts()
            for key in ['0', '1']:
                if key not in counts.keys():
                    counts[key] = 0

            val = (counts['0'] - counts['1']) / float(self.qinstance.run_config.shots)

        self.circuit_counts += 1

        return val

    def _calc_grad_A(self, params, dparams, *, is_shift=False):

        if self.optimize_shift:
            qc = QuantumCircuit(self.qreg, self.qreg_ancilla, self.inc_ancilla)
        else:
            qc = QuantumCircuit(self.qreg, self.qreg_ancilla)

        qc = self.state_preparation(qc, zero_state='grad_ansatz', one_state='ansatz', params=params, dparams=dparams)

        if is_shift:
            qc = self.shift_add(qc)

        qc.h(self.qreg_ancilla)
        qc.h(self.qreg[0])
        if self.qinstance.is_statevector:
            sv = self.qinstance.execute(qc).get_statevector(qc)
            val = 0

            len_sv = len(sv)
            qc_num = (int) (qc.num_qubits)

            if self.optimize_shift: # last n - 2 qubits do not contribute to overall result of simulation
                qc_num = (int) (qc_num/2 + 1) # get number of qubits from n + (n - 2) = qc_num
                inc_ancilla = (int) (qc_num - 2)
                len_sv = (int) (len_sv >> inc_ancilla)

            for l in range(len_sv):
                bits = bin(l)[2:].zfill(qc_num)
                if bits[self.num_mct_ancilla] == '0' and bits[-1] == '0':
                    val += np.real(sv[l]*sv[l].conjugate())
                elif bits[self.num_mct_ancilla] == '1' and bits[-1] == '0':
                    val -= np.real(sv[l]*sv[l].conjugate())
                elif bits[self.num_mct_ancilla] == '0' and bits[-1] == '1':
                    val -= np.real(sv[l]*sv[l].conjugate())
                elif bits[self.num_mct_ancilla] == '1' and bits[-1] == '1':
                    val += np.real(sv[l]*sv[l].conjugate())
        else:
            creg = ClassicalRegister(2, 'c')
            qc.add_register(creg)
            qc.measure(self.qreg_ancilla, creg[1])
            qc.measure(self.qreg[0], creg[0])
            counts = self.qinstance.execute(qc).get_counts()

            for key in ['00', '01', '10', '11']:
                if not key in counts.keys():
                    counts[key] = 0

            val = (counts['00'] - counts['01'] - counts['10'] + counts['11']) / float(self.qinstance.run_config.shots)

        self.circuit_counts += 1

        return val

    def _calc_grad_for_bc(self, params, dparams, *, is_identity=False):

        if self.optimize_shift:
            qc = QuantumCircuit(self.qreg, self.qreg_ancilla, self.inc_ancilla)
        else:
            qc = QuantumCircuit(self.qreg, self.qreg_ancilla)

        qc = self.state_preparation(qc, zero_state='grad_ansatz', one_state='ansatz', params=params, dparams=dparams)
        qc = self.shift_add(qc)

        qc.h(self.qreg_ancilla)
        if not is_identity:
            qc.h(self.qreg[0])

        if self.qinstance.is_statevector:
            sv = self.qinstance.execute(qc).get_statevector(qc)
            val = 0

            len_sv = len(sv)
            qc_num = (int) (qc.num_qubits)

            if self.optimize_shift: # last n - 2 qubits do not contribute to overall result of simulation
                qc_num = (int) (qc_num/2 + 1) # get number of qubits from n + (n - 2) = qc_num
                inc_ancilla = (int) (qc_num - 2)
                len_sv = (int) (len_sv >> inc_ancilla)

            for l in range(len_sv):
                bits = bin(l)[2:].zfill(qc_num)
                if is_identity:
                    if bits[self.num_mct_ancilla] == '0' and not np.array([int(bits[i]) for i in range(self.num_mct_ancilla+1, len(bits)-1)]).any():
                        val += np.real(sv[l]*sv[l].conjugate())
                    elif bits[self.num_mct_ancilla] == '1' and not np.array([int(bits[i]) for i in range(self.num_mct_ancilla+1, len(bits)-1)]).any():
                        val -= np.real(sv[l]*sv[l].conjugate())
                else:
                    if not np.array([int(bits[i]) for i in range(self.num_mct_ancilla+1, len(bits)-1)]).any():
                        if bits[self.num_mct_ancilla] == '0' and bits[-1] == '0':
                            val += np.real(sv[l]*sv[l].conjugate())
                        elif bits[self.num_mct_ancilla] == '0' and bits[-1] == '1':
                            val -= np.real(sv[l]*sv[l].conjugate())
                        elif bits[self.num_mct_ancilla] == '1' and bits[-1] == '0':
                            val -= np.real(sv[l]*sv[l].conjugate())
                        elif bits[self.num_mct_ancilla] == '1' and bits[-1] == '1':
                            val += np.real(sv[l]*sv[l].conjugate())
        else:
            creg = ClassicalRegister(self.num_qubits+1, 'c')
            qc.add_register(creg)
            qc.measure(self.qreg_ancilla, creg[-1])
            qc.measure(self.qreg, creg[:-1])
            counts = self.qinstance.execute(qc).get_counts()

            for key in ['0'+'0'*(self.num_qubits-1)+'0', '0'+'0'*(self.num_qubits-1)+'1', '1'+'0'*(self.num_qubits-1)+'0', '1'+'0'*(self.num_qubits-1)+'1']:
                if key not in counts.keys():
                    counts[key] = 0

            if is_identity:
                val = (counts['0'+'0'*(self.num_qubits-1)+'0'] + counts['0'+'0'*(self.num_qubits-1)+'1'] - counts['1'+'0'*(self.num_qubits-1)+'0'] - counts['1'+'0'*(self.num_qubits-1)+'1']) / float(self.qinstance.run_config.shots)
            else:
                val = (counts['0'+'0'*(self.num_qubits-1)+'0'] - counts['0'+'0'*(self.num_qubits-1)+'1'] - counts['1'+'0'*(self.num_qubits-1)+'0'] + counts['1'+'0'*(self.num_qubits-1)+'1']) / float(self.qinstance.run_config.shots)

        self.circuit_counts += 1

        return val

    def get_A_matrix(self):

        I0 = np.array([[1, 0], [0, 0]])
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])

        P = np.zeros((2**self.num_qubits, 2**self.num_qubits))
        for i in range(2**self.num_qubits):
            P[(i+1)%(2**self.num_qubits), i] = 1

        A0 = I - X
        for i in range(self.num_qubits-1):
            A0 = np.kron(I, A0)
        A1 = P.T @ A0 @ P

        if self.bc == 'Periodic':
            B = 0
            c = self.c
        elif self.bc == 'Neumann':
            B0 = I - X
            for i in range(self.num_qubits-1):
                B0 = np.kron(I0, B0)
            B = P.T @ B0 @ P
            c = self.c
        elif self.bc == 'Dirichlet':
            B0 = - X
            for i in range(self.num_qubits-1):
                B0 = np.kron(I0, B0)
            B = P.T @ B0 @ P
            c = 0
        else:
            raise ValueError('Invalid boundary condition setting. Boundary condition is either "Periodic", "Neumann", or "Dirichlet".')

        A = A0 + A1 - B + c*np.eye(2**self.num_qubits)

        return A

    def get_f_vec(self):
        backend = Aer.get_backend('statevector_simulator')
        return execute(self.qc_f_vec, backend).result().get_statevector()

    def get_cl_sol(self):
        return np.linalg.inv(self.get_A_matrix()) @ np.array(self.get_f_vec())

    def minimize(self, x0, *, method=None, bounds=None, constraints=(), tol=None, options=None, use_grad=True, save_logs=False):

        self.objective_counts = 0
        self.circuit_counts = 0
        self.objective_logs = []
        self.error_logs = {}
        self.objective_count_logs = []
        self.circuit_count_logs = []
        self.sol_logs = []

        if use_grad:
            jac = self.grad
        else:
            jac = None

        if save_logs:
            callback = self._callback
        else:
            callback = None

        if method == 'spsa':
            print("spsa")
            # create Qiskit SPSA solver
            spsa = SPSA(maxiter=2000, callback=self._callback_spsa)
            out = spsa.minimize(self.objective, x0, jac=jac, bounds=bounds)
            res = {'x': out.x}
        else:    
            res = minimize(self.objective, x0, method=method, jac=jac, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)

        self.res = res

        return res

    def _callback(self, xk):

        self.objective_logs.append(self.current_objective)
        self.objective_count_logs.append(self.objective_counts)
        self.circuit_count_logs.append(self.circuit_counts)
        self.sol_logs.append(xk)
        err = self.get_errors(xk)

        for key in err.keys():
            if key not in self.error_logs.keys():
                self.error_logs[key] = [err[key]]
            else:
                self.error_logs[key].append(err[key])

        if 'tolfun' in dir(self):
            print('It.: %05d, Obj.: %.6e, Tolfun.: %.6e'%(len(self.objective_logs), self.current_objective, self.tolfun(xk)))
        else:
            print('It.: %05d, Obj.: %.6e'%(len(self.objective_logs), self.current_objective))
    
    def _callback_spsa(self, iter, xk, feval, step, step_accepted):

        self.objective_logs.append(self.current_objective)
        self.objective_count_logs.append(self.objective_counts)
        self.circuit_count_logs.append(self.circuit_counts)
        self.sol_logs.append(xk)
        err = self.get_errors(xk)

        for key in err.keys():
            if key not in self.error_logs.keys():
                self.error_logs[key] = [err[key]]
            else:
                self.error_logs[key].append(err[key])

        if 'tolfun' in dir(self):
            print('It.: %05d, Obj.: %.6e, Tolfun.: %.6e, Iter.: %05d'%(len(self.objective_logs), self.current_objective, self.tolfun(xk), iter))
        else:
            print('It.: %05d, Obj.: %.6e, Iter.: %05d'%(len(self.objective_logs), self.current_objective, iter))

    def get_statevec(self, x):

        qc = QuantumCircuit(self.qreg)
        qc = self.ansatz(qc, x)

        backend = Aer.get_backend('statevector_simulator')

        return execute(qc, backend).result().get_statevector()

    def print_matrix(self, x):
        qc = QuantumCircuit(self.qreg)
        qc = self.ansatz(qc, x)

        # res = execute(qc, Aer.get_backend('statevector_simulator')).result()
        # print(res.get_unitary(qc, decimals=3))
        print(qi.Operator(qc).data)
        return

    def get_errors(self, x):

        statevec = self.get_statevec(x)
        solvec = np.array(self.evaluate(x)[1]*statevec)

        cl_sol = self.get_cl_sol()
        cl_sol_normalized = cl_sol / np.linalg.norm(cl_sol)
        cl_dot_state = np.vdot(cl_sol_normalized, statevec)

        err = {}
        err['trace'] = np.sqrt( 1 - np.real(cl_dot_state.conjugate()*cl_dot_state) )
        err['relative'] = np.linalg.norm(cl_sol-solvec) / np.linalg.norm(cl_sol)

        return err

    def get_sol(self, x):
        return self.evaluate(x)[1]*self.get_statevec(x)