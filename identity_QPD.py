import numpy as np
from helper_gates import ymeas, xmeas, zmeas, idmeas

#define the identity channel QPD
identity_QPD = [
    {"op": idmeas, "rho":[1,0], "c": 1},
    {"op": idmeas, "rho":[0,1], "c": 1},
    {"op": xmeas, "rho":[1,1]*1/np.sqrt(2), "c": 1},
    {"op": xmeas, "rho":[1,-1]*1/np.sqrt(2), "c": -1},
    {"op": ymeas, "rho":[1,1j*1]*1/np.sqrt(2), "c": 1},
    {"op": ymeas, "rho":[1,-1j*1]*1/np.sqrt(2), "c": -1},
    {"op": zmeas, "rho":[1,0], "c": 1},
    {"op": zmeas, "rho":[0,1], "c": -1}
]