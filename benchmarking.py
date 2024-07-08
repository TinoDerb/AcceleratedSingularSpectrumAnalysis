import numpy as np
from SSA import SSA
import time
# data size
numberOfSamples = [1000,1000,10000,100000,1000000,10000000,100000000]
# benchmarking accelerated
print("Accelerated")
for n in numberOfSamples:
    signal = np.random.rand(n)
    startTime = time.time()
    RC = SSA(signal, 10, 10, "accelerated")
    print(str(n)+"\t\t"+str(time.time()-startTime))
# benchmarking classic, remember to start over
print("Classical")
for n in numberOfSamples:
    signal = np.random.rand(n)
    startTime = time.time()
    RC = SSA(signal, 10, 10, "classic")
    print(str(n)+"\t\t"+str(time.time()-startTime))