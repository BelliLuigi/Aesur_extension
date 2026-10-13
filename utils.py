import numpy as np

def separate_phase_ampl(dataset):
    ampl = abs(dataset)
    phase_non = -1*np.unwrap(np.angle(dataset))/100
    out = phase_non
    phase= out-out[..., np.newaxis,0]
    return [ampl, phase]