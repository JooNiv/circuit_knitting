import numpy as np

#used to convert measurement outcomes from strings to numpy arrays
def list_to_numpy_array(meas_res, init_res, order):
    if order == ("meas", "init"):
        res = meas_res[1::][::-1] + init_res[::-1]
        return np.array(list(res)).astype(int)
    else:
        res = init_res[::-1] + meas_res[:-1][::-1]
        return np.array(list(res)).astype(int)

#convert measurement outcome strings and do classical post processing [0,1] -> {-1,1}
def to_numpy_exp(meas_res, init_res, order):
    arr = list_to_numpy_array(meas_res[0], init_res[0], order)
    result = np.array([list(map(lambda x: -1 if x == 0 else 1, arr))])[0]
    return result

#prints the two subcircuits in order
def print_in_order(circs):
    print(circs[circs["order"][0]])
    print(circs[circs["order"][1]])

#calculate relative error
def relative_error(actual, approx):
    if(np.prod(actual) == 0):
        return abs(approx-actual)/(1+abs(actual))
    else:
        return abs(approx-actual)/(abs(actual))
