import sys
import json

import numpy as np

from sklearn.svm import SVC 

from qiskit import QuantumCircuit, execute
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel

from qiskit.providers.ibmq.runtime import UserMessenger, ProgramBackend


def program(backend: ProgramBackend, user_messenger: UserMessenger, **kwargs):
    """Function that does SVC using a quantum kernel.
    
    Args:
        backend: Backend for the circuits to run on.
        user_messenger: Used to communicate with the program user.
        kwargs: User inputs.
                kz_kernel: Quantum kernel
                kX_tr: List of input training data
                kY_tr: List of target training data
                kX_va: List of input validation data
                kY_va: List of target validation data
                
    Returns: Model accuracy score on validation data
    """  
    # get data
    z_kernel = kwargs.pop('kz_kernel', [])
    X_tr = kwargs.pop('kX_tr', [])
    Y_tr = kwargs.pop('kY_tr', [])
    X_va = kwargs.pop('kX_va', [])
    Y_va = kwargs.pop('kY_va', [])
    
    # do SVC
    q_model = SVC(kernel=z_kernel.evaluate)
    q_model.fit(X_tr, Y_tr)
    q_score = q_model.score(X_va, Y_va)
    
    return q_score


def main(backend: ProgramBackend, user_messenger: UserMessenger, **kwargs):
    """Main entry point of runtime program.

    Args:
        backend: Backend for the circuits to run on.
        user_messenger: Used to communicate with the program user.
        kwargs: User inputs.
                X_tr: List of input training data
                Y_tr: List of target training data
                X_va: List of input validation data
                Y_va: List of target validation data
                
    Returns: Result from calling Program
    """
    # set seed
    SEED = 102855

    # set number of shots
    shots = 8092

    # create quantum instance
    qinst = QuantumInstance(backend, shots, SEED)
    
    # encode data via ZZ feature map
    map_z = ZZFeatureMap(feature_dimension=5, reps=1, entanglement='linear')
    
    # create kernel circuit
    z_kernel = QuantumKernel(feature_map=map_z, quantum_instance=qinst)
    
    # get data
    X_tr = kwargs.pop('X_tr', [])
    Y_tr = kwargs.pop('Y_tr', [])
    X_va = kwargs.pop('X_va', [])
    Y_va = kwargs.pop('Y_va', [])
    
    # get score
    result = program(backend, user_messenger, kz_kernel=z_kernel, 
                     kX_tr=X_tr, kY_tr=Y_tr, kX_va=X_va, kY_va=Y_va)
    
    return result

