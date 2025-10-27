# tmm.py - vectorized 2x2 TMM
import numpy as np

def _fresnel_rs_ts(n1, n2, cos1, cos2, pol='s'):
    if pol.lower() == 's':
        r = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        t = (2 * n1 * cos1) / (n1 * cos1 + n2 * cos2)
    else:
        # p-polarization
        r = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        t = (2 * n1 * cos1) / (n2 * cos1 + n1 * cos2)
    return r, t

def tmm_spectrum(wl_nm, n_stack, d_list_nm, theta_inc_deg=0.0, pol='s'):
    wl = np.asarray(wl_nm, dtype=float)
    M = wl.size
    N_layers = len(n_stack)
    # build n_mat: shape N_layers x M
    n_mat = np.zeros((N_layers, M), dtype=complex)
    for i in range(N_layers):
        arr = n_stack[i]
        if np.isscalar(arr) or arr.shape == ():
            n_mat[i,:] = arr
        else:
            n_mat[i,:] = np.asarray(arr, dtype=complex)
    # Snell law
    theta0 = np.deg2rad(theta_inc_deg)
    sin_t0 = np.sin(theta0)
    sin_t_i = (n_mat[0,:]*sin_t0) / n_mat
    cos_t_i = np.sqrt(1 - sin_t_i**2 + 0j)
    k0 = 2*np.pi / (wl*1e-9)
    kz = k0[np.newaxis,:] * n_mat * cos_t_i
    # interface fresnel
    r_if = np.zeros((N_layers-1, M), dtype=complex)
    t_if = np.zeros((N_layers-1, M), dtype=complex)
    for i in range(N_layers-1):
        r_if[i,:], t_if[i,:] = _fresnel_rs_ts(n_mat[i,:], n_mat[i+1,:], cos_t_i[i,:], cos_t_i[i+1,:], pol=pol)
    # check d list
    if len(d_list_nm) != max(0, N_layers-2):
        if len(d_list_nm) == 0 and N_layers-2==0:
            pass
        else:
            raise ValueError("d_list length must be N_layers-2")
    R = np.zeros(M, dtype=float)
    T = np.zeros(M, dtype=float)
    for idx in range(M):
        Mtot = np.eye(2, dtype=complex)
        for j in range(N_layers-1):
            tij = t_if[j,idx]; rij = r_if[j,idx]
            D = (1.0/tij) * np.array([[1, rij],[rij, 1]], dtype=complex)
            Mtot = Mtot @ D
            if 1 <= j <= N_layers-2:
                d_nm = d_list_nm[j-1]
                phase = np.exp(1j * kz[j, idx] * (d_nm*1e-9))
                P = np.array([[phase, 0],[0, 1/phase]], dtype=complex)
                Mtot = Mtot @ P
        r_tot = Mtot[1,0] / Mtot[0,0]
        t_tot = 1.0 / Mtot[0,0]
        R[idx] = np.abs(r_tot)**2
        factor = (n_mat[-1, idx].real * np.real(cos_t_i[-1, idx])) / (n_mat[0, idx].real * np.real(cos_t_i[0, idx]))
        T[idx] = factor * (np.abs(t_tot)**2)
    A = 1.0 - R - T
    R = np.clip(R, 0.0, 1.0)
    T = np.clip(T, 0.0, 1.0)
    A = np.clip(A, 0.0, 1.0)
    return R, T, A
