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
