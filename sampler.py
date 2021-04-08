__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import sys
import os
import numpy as np
import numpy.linalg as la
import pickle

EPSILON = 1e-6
rnd_seed = 12345
np.random.seed(rnd_seed)

from numpy import linalg as la

#-------------------------------------------------------------------------------
def norm(a):
    return a / la.norm(a)

#-------------------------------------------------------------------------------
def todeg(x):
    deg = np.arccos(x)*180/np.pi
    s = str(deg.round(4))+'°'
    return s

#-------------------------------------------------------------------------------
def nearestPDX(A):
    A3 = nearcorr(A)
    A3 = A3+np.eye(A3.shape[0])*1e-8
    L = chol(A3)
    if L is None:
        print('Cannot make H PSD !!')
        sys.exit(0)
    return A3, L

#-------------------------------------------------------------------------------
def nearestPD(A):
    # From https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite?rq=1
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    L = chol(A3)
    if L is not None:
        return A3, L

    print("Still H is not PSD, fixing it...")
    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while True:
        L = chol(A3)
        if L is not None:
            break

        print("Increasing eigen values...")
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    print("H is PSD now")
    return A3, L

#-------------------------------------------------------------------------------
def chol(B):
    try:
        L = la.cholesky(B)
        return L
    except la.LinAlgError:
        return None

#--------------------------------------------------------------------
def get_covb(z):
    print("Zeros in latents: ",(z.sum(0) == 0).sum())
    n = z.shape[0]
    x = z
    if z.min() == 0 and z.max() == 1:
        x = (z*2)-1
    
    xmu = x.sum(0)/n
    p = (xmu+1)/2

    D = (x.T @ x)/n
    if chol(D) is not None:
        print("D is PSD")

    G = D
    G = np.vstack([D, xmu])
    G = np.hstack([G, np.append(xmu, [1])[:,None]])
        
    if chol(G) is not None:
        print("G is PSD")

    H = G
    H = np.sin(G * np.pi/2)
    L = chol(H)
    eval, evec = la.eig(H)

    if L is None:
        print("*** H is  NOT PSD")
        print("Neg evals:", (eval<0).sum()," min eval:", eval.min())
        Hs = H 
        H,L = nearestPD(H)
        print("H  diff:", np.abs(Hs-H).sum())
    else:
        print("OK - H IS PSD")

    return L, xmu, [H, eval]

#--------------------------------------------------------------------
def sample_boundary(L, mu, B, neg_zero = False, zsize=None, ref=None, attr=0):
    if zsize is None:
        zn = mu.shape[0] if mu is not None else L.shape[0]
    else:
        zn = zsize

    zr = np.random.randn(B, L.shape[0])
    zr = zr / la.norm(zr, axis=-1)[:,None]

    if ref is not None:
        ref = norm(ref)
        zr =  zr + .3*ref
        zr = norm(zr)
    
    a = zr @ L.T
    b = np.sign(a)
    b[b==0]=1

    v = b
    if L.shape[0] != zn:
        b = b*b[:, -1][:,None]
        if zn > b.shape[1]:
            b = np.hstack([b, -np.ones((b.shape[0], zn-b.shape[1]))])
        v = b[:,:zn]

    return v

sample = sample_boundary

#--------------------------------------------------------------------
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high 
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

#--------------------------------------------------------------------
def sample_cut(L, m):
    a = L @ m
    b = np.sign(a)
    b[b==0]=1
    bc = b*b[-1]
    zn = b.shape[0]-1
    return b[:zn], bc[:zn], b[-1]

#--------------------------------------------------------------------
def split_sphere(L, l1, l2=None, m2=None, verbose=False):
    # Get mean vectors represnting the positive and negative vectors in the sphere
    mv = 0
    z1 = np.append(l1, [mv])
    if l2 is not None:
        z2 = np.append(l2, [mv])

    lone = L[z1==1]
    lzero = L[z1==-1]
    p = norm(lone.mean(0))
    n = norm(lzero.mean(0))

    #-----------------------------
    bp = None
    bp_n = 10000
    while True:
        dp = lone @ p
        zv = dp[dp<0]
        inv = zv.shape[0]
        if verbose: print('pd:',inv)
        if bp_n > inv:
            bp_n = inv
            bp = p
        else:
            break

        if inv == 0:
            break

        lone = np.concatenate([lone, lone[dp<0]])
        p = norm(lone.mean(0))
    p = bp 

    #-----------------------------
    bn = None
    bn_n = 10000
    for i in range(10):
        dn = lzero @ n
        zv = dn[dn<0]
        inv = zv.shape[0]
        if verbose: print('nd:',inv)
        if inv <= bn_n:
            bn_n = inv
            bn = n
        else:
            break

        if inv == 0:
            break

        lzero = np.concatenate([lzero, lzero[dn<0]])
        n = norm(lzero.mean(0))
    n = bn 

    #-----------------------------
    dt = n.T @ p[:,None]
    if verbose: print('p.n', dt, ' theta:', todeg(dt))
    
    m = (p-n)
    m = m / la.norm(m)
    m = m[:,None]
    c = m

    dt = L[-1,:] @ c
    if verbose: print('mean.c', dt, ' theta:', todeg(dt))
    dt = L[-1,:] @ m
    if verbose: print('mean.m', dt, ' theta:', todeg(dt))
    dt = m.T @ p[:,None]
    if verbose: print('m.p', dt, ' theta:', todeg(dt))
    dt = m.T @ n[:,None]
    if verbose: print('m.n', dt, ' theta:', todeg(dt))

    # Sample with the new vector
    #-----------------------------
    b,bc,_ = sample_cut(L, m)
    hd = (b != l1[:,None]).sum()
    hdc = (bc != l1[:,None]).sum()
    if verbose: print(hd, hdc,'hamming dist raw/mean corrected')

    return m[:,0], b[:,0], c[:,0]

#--------------------------------------------------------------------
def interpolate(L, begin, end, steps, verbose=False, set_attr=-1):
    # Set attribute
    if set_attr>-1 and os.path.isfile('prob_dim.pk'):
        # Interpolate towards attribute with mean stored in prob_dim.pk
        # This file is generated in sampler_test.py in show_classes()
        end = begin.copy()
        cor_pos, cor_prob = pickle.load(open('prob_dim.pk', 'rb'))    
        T = 0.12
        pos_idx = cor_prob > T
        pos_dims = cor_pos[pos_idx]

        # Add attribute
        end[pos_dims] = 1
        neg_idx = cor_prob < -T
        neg_dims = cor_pos[neg_idx]

        # Add attr
        end[neg_dims] = -1
        # print("pos/neg dims shape:", pos_dims.shape, ",  ",neg_dims.shape)

    if verbose: print("INTERPOLATION ---------------------------------")
    # Get vector in the covariances unit sphere for begin and end
    bm ,bz, cb = split_sphere(L, begin, verbose)
    if verbose: print("---------------------------------")
    em, ez, ce = split_sphere(L, end, begin, bm, verbose)
    if verbose: print('bm -> em:', todeg(bm.T @ em), '  Hamm dist:',(bz != ez).sum())

    out = [begin]
    hdists = []
    for i in range(steps):
        step = i/(steps-1)
        v = slerp(step, bm, em)
        b,bc,_ = sample_cut(L, v)
        if verbose: 
            print('2 bm -> v'+str(i)+':', todeg(bm.T @ v), '  Hamm dist:',(bz!=b).sum(), ' - ',(ez!=b).sum())
        hdists.append([(bz!=b).sum(), (ez!=b).sum()])
        out.append(b)

    out.append(end)
    return np.array(out), np.array(hdists)

#--------------------------------------------------------------------
def interpolate_rnd(d, L, B, steps, verbose=False, labels=[], set_attr=-1):
    try:
        set_attr = int(set_attr)
    except:
        print(f"The attribute must be an integer number. Given attribute is {set_attr}.")
        print(f"Ignoring this attribute")
        set_attr = -1

    if len(labels)  == 0:
        set_attr = -1

    if set_attr>-1:
        set_attr_stats(d, labels, set_attr)

    src = np.random.choice(d.shape[0],(B,2),replace=False)
    out = []
    hdists = []
    for bs in range(src.shape[0]):
        ips, dists = interpolate(L, d[src[bs,0]], d[src[bs,1]], steps, verbose=verbose, set_attr=set_attr)
        out.append(ips)
        hdists.append(dists)

    out =np.array(out) 
    out = out.reshape(-1, d.shape[1])
    hdists = np.array(hdists)
    return out, hdists


#--------------------------------------------------------------------
def set_attr_stats(latents, labels, attr):
    d=latents
    c = attr
    print(f"Setting attribute: {attr}")

    if labels.shape[1] == 40:
        idx = labels[:, c] == 1
    else:
        idx = labels == c
        idx = idx[:,0]

    dc = d[idx]

    # Stored sorted mean for the above distribution 
    cor_mean = dc.mean(axis=0)
    cor_sorted = np.argsort(cor_mean)
    cor_mean_sorted = cor_mean[cor_sorted]
    
    filename_prob = 'prob_dim.pk'
    print(f"Latent stats saved in {filename_prob}")
    pickle.dump([cor_sorted, cor_mean_sorted], open(filename_prob, 'wb'))
    return

#--------------------------------------------------------------------
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")

