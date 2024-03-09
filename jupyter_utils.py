import numpy as np

def findbin(h_axis, value): # inclusive on bottom bin, exclusive on top bin
    edges = h_axis.edges()
    for i in range(len(edges)-1):
        if edges[i] <= value and value < edges[i+1]:
            return i
    return len(edges)-1

def projectionXY(h):
    return np.sum(h, axis=2)

def xslice(h, bin_lo, bin_hi): # input can be any dimension
    h_slice = h[bin_lo:bin_hi]
    return np.sum(h_slice, axis=0)

def yslice(th2d, bin_lo, bin_hi): # input must be 2D np array
    h_slice = th2d.T[bin_lo:bin_hi]
    return np.sum(h_slice, axis=0)

def zslice(th3d, bin_lo, bin_hi): # input must be 3D np array
    h_slice = th3d[:, :, bin_lo:bin_hi]
    return np.sum(h_slice, axis=2)

def projectionX(h):
    if len(h.shape) == 3:
        return np.sum(projectionXY(h), axis=1)
    return np.sum(h, axis=1)

def projectionY(h):
    if len(h.shape) == 3:
        return np.sum(projectionXY(h), axis=0)
    return np.sum(h, axis=0)

def projectionZ(h):
    return np.sum(np.sum(h, axis=0), axis=0)

def rebin(h, bin_edges, factor):
    assert len(h) % factor == 0
    bin_width = bin_edges[1] - bin_edges[0]
    h_combined = np.array([np.sum(h[i*factor:(i+1)*factor]) for i in range(int(len(h) / factor))]) / factor
    bin_edges = bin_edges[::factor]
    return h_combined, bin_edges

def get_log_bincenters(logbins):
    return [np.exp( (np.log(logbins[i+1]) + np.log(logbins[i])) / 2 ) for i in range(len(logbins)-1)]

def get_lin_bincenters(linbins):
    return [(linbins[i+1] + linbins[i]) / 2 for i in range(len(linbins)-1)]

def safe_divide(a, b, filler=0):
    if filler == 1:
        return np.divide(a, b, out=np.ones_like(a), where=b!=0)
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def get_ratio_safe(num, denom, numerr=None, denomerr=None, filler=0):
    assert len(num) == len(denom)

    central_vals = safe_divide(num, denom, filler=filler)
    if numerr is None: numerr = np.zeros(len(num))
    if denomerr is None: denomerr = np.zeros(len(denom))

    assert len(denom) == len(numerr) and len(numerr) == len(denomerr)
    errs = np.sqrt( safe_divide(numerr, denom)**2 + safe_divide(num * denomerr, denom**2)**2 )
    return central_vals, errs

def get_ratio_err(num, denom, numerr, denomerr):
    return np.sqrt( (numerr / denom)**2 + (num * denomerr / (denom**2))**2 )

def get_binwidths(a):
    return [a[i+1] - a[i] for i in range(len(a) - 1)]

def get_err_ab(a, b, a_err=None, b_err=None):
    if a_err is None: a_err = np.zeros(len(a))
    if b_err is None: b_err = np.zeros(len(b))
    return np.sqrt( (b*a_err)**2 + (a*b_err)**2 )
