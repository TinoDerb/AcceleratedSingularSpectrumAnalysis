import numpy as np
import time
from SSA import SSA

def test_ssa_accelerated():
    # Test the accelerated version of SSA
    signal = np.random.rand(1000)
    start_time = time.time()
    RC = SSA(signal, 10, 10, "accelerated")
    duration = time.time() - start_time
    
    # Example assertions (you should adjust these based on your actual expectations)
    assert isinstance(RC, np.ndarray), "Result should be a numpy array"
    assert RC.shape[0] == len(signal), "Number of rows in RC should match the shape of the input signal"
def test_ssa_classical():
    # Test the classical version of SSA
    signal = np.random.rand(1000)
    start_time = time.time()
    RC = SSA(signal, 10, 10, "classic")
    duration = time.time() - start_time
    
    # Example assertions (you should adjust these based on your actual expectations)
    assert isinstance(RC, np.ndarray), "Result should be a numpy array"    
    assert RC.shape[0] == len(signal), "Number of rows in RC should match the shape of the input signal"
