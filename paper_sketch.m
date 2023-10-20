%% VQA Poisson validation

H = [1 1 ; 1 -1]'/sqrt(2);
X = [0 1; 1 0]';


XH = H * X ;

F = kron(XH, kron(H, kron(H, H)));

u0 = [1 0]';
initial_state = kron(u0, kron(u0, kron(u0,u0)));
f = F * initial_state;

% syms a b;
% assume(a,"real")
% assume(b,"real")
% assume(a^2 + b ^2 == 1);
% 
% uPhi = [a b]';
% uPhiG = kron(uPhi, kron(uPhi, kron(uPhi,uPhi)));
% 
% f = F * uPhiG;

%% 2D generalization
N = 4;
A = diag(2*ones(1,N)) + diag(-1*ones(1,N-1),1) + diag(-1*ones(1,N-1),-1);
A(1,N) = -1;
A(N,1) = -1;

A2 = kron(A, eye(4)) + kron(eye(4),A);

% optimal circuit values from qiskit implementation
% 0's are just to make the entry the same dimension, it is ignored
params_init = [ [6.896593888515826 8.987334637284528 7.574547976478632 6.847204019070697 0 0] 
                [5.323803221044644 8.116544802428407 5.498883072830878 11.206350031706826 12.109743395712375 4.8184682344975025] 
                [9.949110253214393 6.646289577666751 7.1382584801773445 11.63139039619053 0.8926654342978345 1.0948990714191347] 
                [0.2540718754635438 10.463009560025881 9.77860612850141 10.932895093864325 12.297680778506693 10.042522697586502]
                [5.799120696948454 9.808418904536085 1.4862802696695647 8.041484717920905 1.8014305383688254 11.871059719510393] 
                [6.557739415592751 5.210795617590082 3.324503869815388 9.729307483553026 5.73215413049621 7.143151671267129] ];

params_optimal = [[6.38264276  8.71081761  7.20155899  6.68731467 0 0]
                  [4.84059749  7.01263258 4.92084157 10.53187878 12.54619094  4.06856664] 
                  [10.98972976  6.50613609 7.69234512 12.11597324  1.1721177   0.59477625]
                  [0.38148111 10.88287468 10.39659323 10.1022631  12.42411446  9.58981634]
                  [5.85799763  8.96085645 1.49836136  8.02503515  2.6708049  12.29937232]
                  [7.16844938  4.87774563 3.5625847  10.69383531  5.77682222  7.404504]
                  ];

qc = ansatz(params_init, N);

    % 
    % def ansatz(self, qc, params, *, control=None):
    % 
    %     params = [params[:self.num_qubits]] \
    %             + [params[self.num_qubits+i_layer*self.num_params_per_layer:self.num_qubits+(i_layer+1)*self.num_params_per_layer] \
    %                 for i_layer in range(self.num_layers)]
    % 
    %     if control is None:
    %         for i in range (self.num_qubits):
    %             qc.ry(params[0][i], self.qreg[i])
    %         for i_layer in range(self.num_layers):
    %             for i in range(self.num_qubits//2):
    %                 qc.cz(self.qreg[2*i], self.qreg[2*i+1])
    %                 qc.ry(params[i_layer+1][2*i], self.qreg[2*i])
    %                 qc.ry(params[i_layer+1][2*i+1], self.qreg[2*i+1])
    %             for i in range((self.num_qubits-1)//2):
    %                 qc.cz(self.qreg[2*i+1], self.qreg[2*i+2])
    %                 qc.ry(params[i_layer+1][2*(self.num_qubits//2)+2*i], self.qreg[2*i+1])
    %                 qc.ry(params[i_layer+1][2*(self.num_qubits//2)+2*i+1], self.qreg[2*i+2])

%% Helper Functions
function [qc] = ansatz(params, N)
    num_layers = 5;
    % init to empty gate
    qc = [1];
    % initi each qubit with a Ry gate
    for i=1:N
        qc = kron(Ry(params(1,i)),qc);
    end

    % build gates for each layer
    gate_layer = zeros(size(qc));
    for i_layer=1:num_layers
        
        gate1_Ry = [1];
        gate1_Cz = [1];
        % loop through half of the qubits
        for i=1:(floor(N/2))
            gate1_Cz = kron(Cz(), gate1_Cz);
            gate1_Ry = kron(Ry(params(i_layer+1,2*(i-1) + 1)), gate1_Ry);
            gate1_Ry = kron(Ry(params(i_layer+1,2*(i-1) + 2)), gate1_Ry);
        end
    
        gate2_Ry = [1];
        gate2_Cz = [1];
        % loop through other half
        for i=1:(floor((N-1)/2))
            gate2_Cz = kron(Cz(), gate2_Cz);
            gate2_Ry = kron(Ry(params(i_layer+1,2*floor(N/2)+2*(i-1) + 1)), gate2_Ry);
            gate2_Ry = kron(Ry(params(i_layer+1,2*floor(N/2)+2*(i-1) + 2)), gate2_Ry);
        end
        % add I to make all matrix the same size
        padding = floor(N/4);
        for i = 1:padding
            gate2_Cz = kron(eye(2), kron(gate2_Cz, eye(2)));
            gate2_Ry = kron(eye(2), kron(gate2_Ry, eye(2)));
        end
    
        gate_layer = gate_layer + gate1_Cz + gate1_Ry + gate2_Cz + gate2_Ry;
    end
    
    qc = qc + gate_layer;
end

function [ry] = Ry(theta)
    ry = [cos(theta/2) -sin(theta/2);
          sin(theta/2) cos(theta/2)];
end

function [cz] = Cz()
    cz = eye(4);
    cz(4,4) = -1;
end
