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
    """Function that does SVC using a quantum kernel."""
    
    # get data
    X_tr = kwargs.pop('X_tr', [])
    Y_tr = kwargs.pop('Y_tr', [])
    X_va = kwargs.pop('X_va', [])
    Y_va = kwargs.pop('Y_va', [])
    
    # do SVC
    q_model = SVC(kernel=z_kernel.evaluate)
    q_model.fit(X_tr, Y_tr)
    q_score = q_model.score(X_va, Y_va)
    
    return q_score


def main(backend: ProgramBackend, user_messenger: UserMessenger, **kwargs):
    """This is the main entry point of a runtime program.

    The name of this method must not change. It also must have ``backend``
    and ``user_messenger`` as the first two positional arguments.

    Args:
        backend: Backend for the circuits to run on.
        user_messenger: Used to communicate with the program user.
        kwargs: User inputs.
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
    result = program(backend, user_messenger, z_kernel, X_tr, Y_tr, X_va, Y_va)
    
    return result

