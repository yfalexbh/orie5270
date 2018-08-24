from scipy.optimize import minimize
import numpy as np


def rosenbrock_3d(x):
    if len(x) != 3:
        raise Exception('only 3d optimization will be performed')

    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


# optimize the 3d rosenbrock function below
# try 100 times and choose the best run as the global min
global_min = 1e15
argmin_arr = np.zeros(3)
for i in range(100):
    # generate a random initialization
    arr = np.random.uniform(-10, 10, 3)
    opt_result = minimize(rosenbrock_3d, arr, method='BFGS')
    if opt_result.fun < global_min:
        global_min = opt_result.fun
        argmin_arr = opt_result.x

print(argmin_arr)
