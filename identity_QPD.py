from helper_gates import ymeas, xmeas, zmeas, idmeas, zero_init, plus_init, one_init, minus_init, i_plus_init, i_minus_init

identity_QPD2 = [
    {"op": idmeas, "init":zero_init, "c": 1},
    {"op": idmeas, "init":one_init, "c": 1},
    {"op": xmeas, "init":plus_init, "c": 1},
    {"op": xmeas, "init":minus_init, "c": -1},
    {"op": ymeas, "init":i_plus_init, "c": 1},
    {"op": ymeas, "init":i_minus_init, "c": -1},
    {"op": zmeas, "init":zero_init, "c": 1},
    {"op": zmeas, "init":one_init, "c": -1}
]