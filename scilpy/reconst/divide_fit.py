import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


def get_bounds():
    S0 = [0, 10]
    MD = [1e-12, 4e-9]
    V_I = [1e-24, 5e-18]
    V_A = [1e-24, 5e-18]
    lb = [S0[0], MD[0], V_I[0], V_A[0]]
    ub = [S0[1], MD[1], V_I[1], V_A[1]]
    return lb, ub


def random_p0(signal, gtab_infos, lb, ub, weight, n_iter):    
    guess = []
    thr = np.inf

    for i in range(n_iter):
        params_rand = lb + (ub - lb) * np.random.rand(len(lb))
        signal_rand = gamma_fit2data(gtab_infos, params_rand)
        residual_rand = np.sum(((signal - signal_rand) * weight)**2)

        if residual_rand < thr:
            thr = residual_rand
            guess = params_rand

    #guess = np.array([500,1e-9,1e-23,1e-20])
    return guess


def gamma_data2fit(signal, gtab_infos, fit_iters=1, random_iters=50, do_weight_bvals=False, do_weight_pa=False, redo_weight_bvals=False, do_multiple_s0=False):

    if np.sum(gtab_infos[3]) > 0 and do_multiple_s0==True:
        ns = len(np.unique(gtab_infos[3])) - 1
    else:
        ns = 0
    
    unit_to_SI = np.array([np.max(signal), 1e-9, 1e-18, 1e-18])
    unit_to_SI = np.concatenate((unit_to_SI, np.ones(ns)))

    def weight_bvals(sthr, mdthr, wthr):
        bthr = -np.log(sthr) / mdthr
        weight = 0.5 * (1 - erf(wthr * (gtab_infos[0] - bthr) / bthr))
        return weight

    def weight_pa():
        weight = np.sqrt(gtab_infos[2] / np.max(gtab_infos[2]))
        return weight

    def my_gamma_fit2data(gtab_infos, *args):
        params_unit = args
        params_SI = params_unit * unit_to_SI
        signal = gamma_fit2data(gtab_infos, params_SI)
        return signal * weight

    lb_SI, ub_SI = get_bounds()
    lb_SI = np.concatenate((lb_SI, 0.5 * np.ones(ns)))
    ub_SI = np.concatenate((ub_SI, 2.0 * np.ones(ns)))
    lb_SI[0] *= np.max(signal)
    ub_SI[0] *= np.max(signal)

    lb_unit = lb_SI / unit_to_SI
    ub_unit = ub_SI / unit_to_SI

    bounds_unit = ([lb_unit, ub_unit])

    res_thr = np.inf

    for i in range(fit_iters):
        weight = np.ones(len(signal))
        if do_weight_bvals:
            weight *= weight_bvals(0.07, 1e-9, 2)
        if do_weight_pa:
            weight *= weight_pa()

        p0_SI = random_p0(signal, gtab_infos, lb_SI, ub_SI, weight, random_iters)
        #print("STARTING : ", p0_SI)
        p0_unit = p0_SI / unit_to_SI
        params_unit, params_cov = curve_fit(my_gamma_fit2data, gtab_infos, signal * weight, p0=p0_unit,
                                    bounds=bounds_unit, method="trf", ftol=1e-8, xtol=1e-8, gtol=1e-8)#, verbose=2)
        #print("params_unit: ", params_unit * unit_to_SI)

        if redo_weight_bvals:
            weight = weight_bvals(0.07, params_unit[1] * unit_to_SI[1], 2)
            if do_weight_pa:
                weight *= weight_pa()
            
            params_unit, params_cov = curve_fit(my_gamma_fit2data, gtab_infos, signal * weight, p0=params_unit, bounds=bounds_unit, method="trf")
        
        signal_fit = gamma_fit2data(gtab_infos, params_unit * unit_to_SI)
        residual = np.sum(((signal - signal_fit) * weight) ** 2)
        if residual < res_thr:
            res_thr = residual
            params_best = params_unit
            params_cov_best = params_cov

    params_best[0] = params_best[0] * unit_to_SI[0]
    return params_best[0:4]


def gamma_fit2data(gtab_infos, params):
    S0 = params[0]
    MD = params[1]
    V_I = params[2]
    V_A = params[3]
    RS = params[4:] # relative signal
    if len(RS) != 0:
        RS = np.concatenate(([1], RS))
        RS_tile = np.tile(RS, len(gtab_infos[0])).reshape((len(gtab_infos[0]), len(RS)))
        RS_index = np.zeros((len(gtab_infos[0]), len(RS)))
        for i in range(len(gtab_infos[0])):
            j = gtab_infos[3][i]
            RS_index[i][int(j)] = 1
        RS_matrix = RS_tile * RS_index
        SW = S0 * np.sum(RS_matrix, axis=1)
    else:
        SW = S0

    V_D = V_I + V_A * (gtab_infos[1] ** 2)
    signal = SW * ((1 + gtab_infos[0] * V_D / MD) ** (-(MD ** 2) / V_D))

    return signal


def gamma_fit2metrics(params):
    # Only function that takes the full brain!!!
    S0 = params[...,0]
    MD = params[...,1]
    V_I = params[...,2]
    V_A = params[...,3]
    V_T = V_I + V_A
    V_L = 5 / 2. * V_A

    MK_I = 3 * V_I / (MD **2)
    MK_A = 3 * V_A / (MD **2)
    MK_T = 3 * V_T / (MD **2)
    microFA2 = (3/2.) * (V_L / (V_I + V_L + (MD**2)))
    microFA = np.sqrt(microFA2)
    return microFA, MK_I, MK_A, MK_T