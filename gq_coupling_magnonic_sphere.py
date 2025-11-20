"""
magnonic_sphere.py

Quadrature-based magnonic modes in a spherical system with Neumann boundary
conditions and dipolar coupling.

Physics:

    k^2 = (1 / l_ex^2) * (Omega - OmegaH) / OmegaM
    l_ex^2 = A / Ms
    l_ex   = sqrt(A / Ms)
    OmegaH = gamma * H - (1/3) * Ms
    OmegaM = gamma * Ms

Scalar mode:
    v(r,theta,phi) (complex)

Vector mode:
    V = (- 0.5 * v, i 0.5 v, 0)

Normalization:
    Modes are normalized such that

        integral_Ball conj(V) . V dV = 1.

Dipolar kernel:
    K_ij(r,r') = δ_ij / |d|^3 - 3 d_i d_j / |d|^5,
    where d = r - r'.

Convolution:
    W(r) = integral K(r,r') V(r') dV'.

Principal value:
    For |r - r'| < eps_len, the contribution is omitted, i.e. we approximate the
    PV integral over that small ball by zero (standard principal-value treatment).

Coupling:
    For modes (l,m,n) and (l',m',n'):

        C = integral_Ball dV  conj(V_{lmn}(r)) . ( M x W_{l'm'n'}(r) ),

    with M = (0,0,1) and W_{l'm'n'} = (K * V_{l'm'n'}).

Selection rules (enforced):
    - preserves m
    - couples l -> l, l±2n
"""

import math
import numpy as np
from numba import njit, prange

from scipy.special import spherical_jn, sph_harm, roots_legendre
from scipy.optimize import root_scalar

# Optional GPU support via numba.cuda
try:
    from numba import cuda
    _HAVE_CUDA = True
    print("numba.cuda imported")
except Exception:
    _HAVE_CUDA = False


# ======================================================================
# Spherical Bessel j_l and derivative
# ======================================================================

def j_sph(l, x):
    """Spherical Bessel j_l(x)."""
    return spherical_jn(l, x)


def j_sph_prime(l, x):
    """Derivative j_l'(x)."""
    return spherical_jn(l, x, derivative=True)


# ======================================================================
# Neumann roots for j_l'(k r0) = 0
# ======================================================================

def _find_neumann_roots(l, n_roots, x_min=1e-6, step=np.pi, x_max=1e4):
    """
    Find the first n_roots positive roots of j_l'(x) using bracketing + brentq.
    """
    roots = []
    f = lambda z: j_sph_prime(l, z)

    x_left = x_min
    f_left = f(x_left)

    while len(roots) < n_roots and x_left < x_max:
        x_right = x_left + step
        f_right = f(x_right)

        if f_left * f_right < 0.0:
            sol = root_scalar(f, bracket=[x_left, x_right], method="brentq")
            if sol.converged and sol.root > 0:
                root = sol.root
                if not roots or abs(root - roots[-1]) > 1e-8:
                    roots.append(root)
            f_left = f_right
            x_left = x_right
        else:
            f_left = f_right
            x_left = x_right

    if len(roots) < n_roots:
        raise RuntimeError(f"Failed to find {n_roots} Neumann roots for l={l}.")
    return np.array(roots, dtype=np.float64)


def neumann_kappa(l, n_pos):
    """
    n_pos : 1-based index of positive roots of j_l'(kappa)=0 (n_pos >= 1).
    """
    return _find_neumann_roots(l, n_pos)[-1]


# ======================================================================
# Neumann radial normalization and vector normalization
# ======================================================================

def neumann_radial_norm(l, k, r0):
    """
    Radial integral for scalar j_l(k r) modes:

        I = integral_0^{r0} r^2 j_l^2(k r) dr

    For Neumann roots kappa = k r0 with j_l'(kappa)=0:

        I = (r0^3 / 2) * (1 - l(l+1)/kappa^2) * j_l(kappa)^2

    For l=0, k=0 (constant mode), I = r0^3 / 3.
    """
    if l == 0 and abs(k) < 1e-15:
        return r0**3 / 3.0

    kappa = k * r0
    jl = j_sph(l, kappa)
    I = (r0**3 / 2.0) * (1.0 - l*(l+1)/kappa**2) * (jl**2)
    return float(I)


def scalar_norm_constant_for_vector(l, k, r0):
    """
    Choose scalar normalization C such that the *vector* mode

        V = (v, -i v, 0),   v = C j_l(k r) Y_l^m,

    satisfies:

        integral conj(V) . V dV = 1.

    Since conj(V).V = 2 |v|^2, we need:

        2 integral |v|^2 dV = 1  =>  integral |v|^2 dV = 1/2.

    With Y_l^m normalized over the sphere, integral |v|^2 dV = C^2 * I_radial,

    so  C = 1 / sqrt(2 * I_radial).
    """
    I_rad = neumann_radial_norm(l, k, r0)
    return 1.0 / math.sqrt(2.0 * I_rad)


# ======================================================================
# Dispersion relations
# ======================================================================

def calc_omega_parameters(Ms, H, gamma, A):
    """
    Given Ms, H, gamma, A, return:

      OmegaH = gamma * H - (1/3) * Ms
      OmegaM = gamma * Ms
      l_ex^2 = A / Ms
      l_ex   = sqrt(A / Ms)
    """
    OmegaH = gamma * H - (1.0/3.0) * Ms
    OmegaM = gamma * Ms
    l_ex_sq = A / Ms
    l_ex = math.sqrt(l_ex_sq)
    return OmegaH, OmegaM, l_ex_sq, l_ex


def calc_k(r0_nm, l, n, m=None):
    """
    Neumann eigen-wavenumber k_{l,n} in 1/nm.

    Neumann BC: j_l'(k r0) = 0 => j_l'(kappa)=0, kappa root, k=kappa/r0.

    Arguments:
      - r0_nm : sphere radius in nm
      - l     : spherical harmonic degree (>= 0)
      - n     : radial index
               * (l=0, n=0) -> k=0, constant Neumann mode
               * (l=0, n>=1) -> n-th positive root of j_0'(kappa)=0
               * (l>0, n>=1) -> n-th positive root of j_l'(kappa)=0
               * (l>0, n=0)  -> INVALID (raises), use n>=1

      - m     : included for API consistency, but does not affect k (degenerate in m).
    """
    r0 = float(r0_nm)
    if n < 0:
        raise ValueError("n must be >= 0")

    if l == 0 and n == 0:
        return 0.0

    if l > 0 and n == 0:
        raise ValueError("For l>0, n must be >= 1 (only l=0 supports n=0 zero mode).")

    kappa = neumann_kappa(l, n)
    return kappa / r0


def calc_omega(r0_nm, l, n, Ms, H, gamma, A, m=None):
    """
    Calculate Omega for given sphere + mode indices using dispersion:

        k^2 = (1 / l_ex^2) * (Omega - OmegaH) / OmegaM

    with k = Neumann eigenvalue (1/nm), l_ex^2 = A/Ms, and
    OmegaH = gamma*H - Ms/3, OmegaM = gamma*Ms (H and Ms in A/m).
    """
    k = calc_k(r0_nm, l, n, m=m)
    OmegaH, OmegaM, l_ex_sq, l_ex = calc_omega_parameters(Ms, H, gamma, A)
    Omega = OmegaH + OmegaM * (k**2 * l_ex_sq)
    return Omega


# ======================================================================
# Spherical quadrature grid
# ======================================================================

def build_spherical_quadrature_grid(r0_nm,
                                    Nr=20,
                                    Ntheta=20,
                                    Nphi=40):
    """
    Build a spherical tensor-product quadrature grid for the ball of radius r0_nm.

    Radial:  Gauss-Legendre in s in [0,1] with weight s^2, mapped to r = r0*s.
    Polar:   Gauss-Legendre in x = cosθ in [-1,1], θ = arccos(x),
             weights wt integrate sinθ dθ.
    Azimuth: Uniform nodes φ_k in [0,2π), equal weights.

    Volume element:
        dV = r^2 sinθ dr dθ dφ
            = r0^3 * s^2 ds * sinθ dθ dφ

    We incorporate s^2 into radial weights, and sinθ into the polar weights via GL.

    Returns:
      points_nm : (N,3) float array of (x,y,z) in nm
      weights   : (N,) float array of volume weights (nm^3)
    """
    r0 = float(r0_nm)

    # ---- Radial: integral_0^1 s^2 f(s) ds via Gauss-Legendre on [0,1]
    xs, ws = roots_legendre(Nr)   # nodes in [-1,1], weights for integral_{-1}^1
    s = 0.5 * (xs + 1.0)          # map [-1,1] -> [0,1]
    w_s = 0.5 * ws                # ds factor
    # Include s^2 in weight (for r^2 factor in volume)
    w_r = w_s * (s**2)
    r = r0 * s

    # ---- Polar: integral_0^π sinθ f(θ) dθ  via x = cosθ in [-1,1]
    xt, wt = roots_legendre(Ntheta)   # x in [-1,1]
    theta = np.arccos(xt)
    # sinθ dθ = -dx, so GL on x with weights wt already integrates sinθ dθ
    w_theta = wt

    # ---- Azimuth: integral_0^{2π} f(φ) dφ via uniform φ nodes
    k = np.arange(Nphi)
    phi = 2.0 * np.pi * k / Nphi
    w_phi = np.full(Nphi, 2.0*np.pi / Nphi, dtype=np.float64)

    # ---- Tensor product grid
    rr, th, ph = np.meshgrid(r, theta, phi, indexing="ij")
    wr, wth, wph = np.meshgrid(w_r, w_theta, w_phi, indexing="ij")

    x = rr * np.sin(th) * np.cos(ph)
    y = rr * np.sin(th) * np.sin(ph)
    z = rr * np.cos(th)

    points_nm = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    # Volume weights in nm^3:
    weights = (wr * wth * wph).ravel() * (r0**3)

    return points_nm, weights


# ======================================================================
# Mode evaluation: scalar v and vector V on quadrature grid
# ======================================================================

def eval_scalar_mode(points_nm, r0_nm, l, m, n):
    """
    Normalized scalar Neumann mode, including n=0 only for l=0:

        v(r,θ,φ) = C * j_l(k r) Y_l^m(θ,φ)

    Mode is normalized in the *vector* sense via C (see scalar_norm_constant_for_vector).

    k from Neumann condition j_l'(k r0)=0:

      - (l=0, n=0) : k=0 constant mode.
      - (l=0, n>=1): n-th positive root of j_0'(kappa)=0.
      - (l>0, n>=1): n-th positive root of j_l'(kappa)=0.
      - (l>0, n=0) : INVALID (raises); use n>=1.

    Returns:
      v : (N,) complex
      k : float (1/nm)
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    R = np.sqrt(X*X + Y*Y + Z*Z)
    r0 = float(r0_nm)

    if l > 0 and n == 0:
        raise ValueError("For l>0, n=0 is not a valid Neumann eigenmode; use n>=1.")

    if l == 0 and n == 0:
        # constant mode, k=0
        theta = np.arccos(np.divide(Z, R, out=np.zeros_like(R), where=R != 0))
        phi = np.arctan2(Y, X)
        Ang = sph_harm(m, l, phi, theta)  # SciPy: (m,l,phi,theta)
        I_rad = r0**3 / 3.0
        C = 1.0 / math.sqrt(2.0 * I_rad)  # vector-normalized
        v = C * Ang
        return v, 0.0

    # nonzero-k mode
    kappa = neumann_kappa(l, n)
    k = kappa / r0

    theta = np.arccos(np.divide(Z, R, out=np.zeros_like(R), where=R != 0))
    phi = np.arctan2(Y, X)

    Rad = spherical_jn(l, k * R)          # radial
    Ang = sph_harm(m, l, phi, theta)      # angular
    C = scalar_norm_constant_for_vector(l, k, r0)

    v = C * Rad * Ang
    v = np.where(R <= r0 + 1e-12, v, np.nan + 1j*np.nan)
    return v, k


def scalar_to_vector(v):
    """
    Map scalar mode v to complex vector field:

        V = (-0.5 * v, i 0.5 v, 0)
    """
    v = np.asarray(v, dtype=np.complex128)
    N = v.size
    V = np.empty((N, 3), dtype=np.complex128)
    V[:, 0] = -0.5 * v
    V[:, 1] = 0.5 * 1j * v
    V[:, 2] = 0.0 + 0.0j
    return V


def calc_mode_field_quadrature(r0_nm, l, m, n,
                               Nr=20, Ntheta=20, Nphi=40):
    """
    Quadrature-based mode field on a spherical tensor-product grid.

    Builds a deterministic grid + weights, evaluates scalar mode v and vector V.

    Returns:
      points_nm : (N,3)
      weights   : (N,)
      V         : (N,3) complex
      meta      : {'k', 'v', 'vector_norm'}
    """
    pts, wts = build_spherical_quadrature_grid(r0_nm, Nr=Nr, Ntheta=Ntheta, Nphi=Nphi)
    v, k = eval_scalar_mode(pts, r0_nm, l, m, n)
    V = scalar_to_vector(v)

    # Check vector normalization under this quadrature:
    vec_norm = np.dot(np.einsum('ij,ij->i', np.conj(V), V), wts)

    return pts, wts, V, {"k": k, "v": v, "vector_norm": vec_norm}


# ======================================================================
# Kernel convolution (CPU parallel, GPU optional, PV with cutoff)
# ======================================================================

@njit(parallel=True)
def _convolve_kernel_collocated_cpu(P, Vsrc, weights, eps_len):
    """
    CPU-parallel core convolution (Numba, parallel over p):

        W_p = sum{q: |r_p - r_q| >= eps_len} K(p,q) V_q w_q

    Principal value:
      For |r_p - r_q| < eps_len, the contribution is dropped, i.e. we approximate
      the PV integral over that small ball by zero.

    P      : (N,3) float64
    Vsrc   : (N,3) complex128
    weights: (N,) float64
    eps_len: float64 (nm)
    """
    N = P.shape[0]
    W = np.zeros((N, 3), dtype=np.complex128)

    eps2 = eps_len * eps_len

    for p in prange(N):
        px, py, pz = P[p, 0], P[p, 1], P[p, 2]
        wx = 0.0 + 0.0j
        wy = 0.0 + 0.0j
        wz = 0.0 + 0.0j

        for q in range(N):
            dx = px - P[q, 0]
            dy = py - P[q, 1]
            dz = pz - P[q, 2]
            r2 = dx*dx + dy*dy + dz*dz

            # Principal value: skip contributions for |d| < eps_len
            if r2 < eps2 or r2 == 0.0:
                kx = vqx
                ky = vqy
                kz = vqz

                wq = weights[q]
                wx += wq * kx
                wy += wq * ky
                wz += wq * kz
            
            else:
                r = math.sqrt(r2)
                inv_r3 = 1.0 / (r2 * r)
                inv_r5 = 1.0 / (r2 * r2 * r)

                vqx = Vsrc[q, 0]
                vqy = Vsrc[q, 1]
                vqz = Vsrc[q, 2]

                dv = dx * vqx + dy * vqy + dz * vqz  # complex
                factor = 3.0 * inv_r5 * dv

                kx = inv_r3 * vqx - factor * dx
                ky = inv_r3 * vqy - factor * dy
                kz = inv_r3 * vqz - factor * dz

                wq = weights[q]
                wx += wq * kx
                wy += wq * ky
                wz += wq * kz

        W[p, 0] = wx
        W[p, 1] = wy
        W[p, 2] = wz

    return W


if _HAVE_CUDA:
    @cuda.jit
    def _convolve_kernel_collocated_cuda(P, Vsrc, weights, eps_len, W):
        """
        GPU core convolution (Numba CUDA):

            W_p = sum_{q: |r_p - r_q| >= eps_len} K(p,q) V_q w_q

        Principal value: 1 * v if |d| < eps_len.

        Each thread handles one p index.
        """
        p = cuda.grid(1)
        N = P.shape[0]
        if p >= N:
            return

        px = P[p, 0]
        py = P[p, 1]
        pz = P[p, 2]
        eps2 = eps_len * eps_len

        wx = 0.0 + 0.0j
        wy = 0.0 + 0.0j
        wz = 0.0 + 0.0j

        for q in range(N):
            dx = px - P[q, 0]
            dy = py - P[q, 1]
            dz = pz - P[q, 2]
            r2 = dx*dx + dy*dy + dz*dz

            # Principal value: skip contributions for |d| < eps_len
            if r2 < eps2 or r2 == 0.0:
                kx = vqx
                ky = vqy
                kz = vqz

                wq = weights[q]
                wx += wq * kx
                wy += wq * ky
                wz += wq * kz
            
            else:
                r = math.sqrt(r2)
                inv_r3 = 1.0 / (r2 * r)
                inv_r5 = 1.0 / (r2 * r2 * r)

                vqx = Vsrc[q, 0]
                vqy = Vsrc[q, 1]
                vqz = Vsrc[q, 2]

                dv = dx * vqx + dy * vqy + dz * vqz  # complex
                factor = 3.0 * inv_r5 * dv

                kx = inv_r3 * vqx - factor * dx
                ky = inv_r3 * vqy - factor * dy
                kz = inv_r3 * vqz - factor * dz

                wq = weights[q]
                wx += wq * kx
                wy += wq * ky
                wz += wq * kz

        W[p, 0] = wx
        W[p, 1] = wy
        W[p, 2] = wz


def convolve_kernel_collocated(points_nm, V, weights,
                               eps_len=0.0,
                               use_gpu=False,
                               threads_per_block=128):
    """
    Convolution wrapper:

        W_p = integral K(r_p,r') V(r') dV'  =  sum_{q: |r_p-r_q|>=eps_len} K(p,q) V_q w_q

    Principal value: for |r_p - r_q| < eps_len, the contribution is dropped.

    Arguments:
      points_nm : (N,3) float array   - quadrature nodes in nm
      V         : (N,3) complex array - source vector field
      weights   : (N,) float array    - quadrature volume weights in nm^3
      eps_len   : small length (nm) defining the PV exclusion radius.
      use_gpu   : if True and a CUDA device is available, run on GPU.
      threads_per_block : CUDA threads per block (1D).

    Returns:
      W : (N,3) complex array
    """
    P = np.asarray(points_nm, dtype=np.float64)
    Vsrc = np.asarray(V, dtype=np.complex128)
    wts = np.asarray(weights, dtype=np.float64)
    N = P.shape[0]

    if use_gpu and _HAVE_CUDA:
        # GPU path
        dP = cuda.to_device(P)
        dV = cuda.to_device(Vsrc)
        dW = cuda.device_array((N, 3), dtype=np.complex128)
        dw = cuda.to_device(wts)

        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        _convolve_kernel_collocated_cuda[blocks_per_grid, threads_per_block](
            dP, dV, dw, float(eps_len), dW
        )
        W = dW.copy_to_host()
        return W
    else:
        # CPU parallel path
        return _convolve_kernel_collocated_cpu(P, Vsrc, wts, float(eps_len))


# ======================================================================
# Couplings and convolution fields (quadrature)
# ======================================================================

def _selection_rules_ok(l1, m1, l2, m2):
    """
    Dipolar kernel is rank-2: preserves m and couples l -> l, l±2n.
    """
    if m1 != m2:
        return False
    return abs(l1 - l2) % 2 == 0


def cross_M_W(W):
    """
    For M = (0,0,1), compute M x W.

    If W = (Wx, Wy, Wz),
       M x W = ( -Wy, Wx, 0 ).

    W : (N,3) complex
    """
    W = np.asarray(W, dtype=np.complex128)
    MW = np.empty_like(W)
    MW[:, 0] = -W[:, 1]
    MW[:, 1] =  W[:, 0]
    MW[:, 2] =  0.0 + 0.0j
    return MW


def calc_coupling_quadrature(r0_nm,
                             l1, m1, n1,
                             l2, m2, n2,
                             Nr=20, Ntheta=20, Nphi=40,
                             eps_len=0.0,
                             use_gpu=False):
    """
    Quadrature-based coupling:

        C = integral conj(V_{l1 m1 n1}(r)) . ( M x W_{l2 m2 n2}(r) ) dV,

    where:
        V_{lmn} = (v, -i v, 0)   (vector-normalized)
        W_{l2 m2 n2} = K * V_{l2 m2 n2} (dipolar kernel)
        M = (0,0,1).

    Uses a Gauss-Legendre x uniform spherical tensor-product grid inside the ball.

    Selection rules:
      if m1!=m2 or |l1-l2| =/= 2n, n = 0,1,2...  ->  returns 0 exactly.

    eps_len:
      small length added inside |d| defining the PV region; contributions
      from |r-r'| < eps_len are 1.

    use_gpu:
      if True and CUDA is available, convolution is done on GPU.

    Returns:
      C : complex coupling coefficient (dimensionless; multiply by gamma*Ms etc.
          externally to get frequency units).
    """
    if not _selection_rules_ok(l1, m1, l2, m2):
        return 0.0 + 0.0j

    # Shared quadrature grid and weights
    pts, wts = build_spherical_quadrature_grid(r0_nm, Nr=Nr, Ntheta=Ntheta, Nphi=Nphi)

    # Mode 1 and 2 on same grid
    v1, k1 = eval_scalar_mode(pts, r0_nm, l1, m1, n1)
    V1 = scalar_to_vector(v1)
    #V1 /= np.sqrt(mode_vector_norm_quadrature(points_nm=pts, weights=wts, V=V1))

    v2, k2 = eval_scalar_mode(pts, r0_nm, l2, m2, n2)
    V2 = scalar_to_vector(v2)
    #V2 /= np.sqrt(mode_vector_norm_quadrature(points_nm=pts, weights=wts, V=V2))

    # W2 = K * V2
    W2 = convolve_kernel_collocated(pts, V2, wts, eps_len=eps_len, use_gpu=use_gpu)

    # M x W2
    MW2 = cross_M_W(W2)

    # Coupling: integral conj(V1) . (M x W2) dV
    dots = np.einsum('ij,ij->i', np.conj(V1), MW2)
    C = np.dot(dots, wts)
    return C


def calc_convolution_field_quadrature(r0_nm, l, m, n,
                                      Nr=20, Ntheta=20, Nphi=40,
                                      eps_len=0.0,
                                      use_gpu=False):
    """
    Quadrature-based convolution field:

        W(r) = integral K(r,r') V_{lmn}(r') dV',

    evaluated at the same quadrature nodes r.

    Principal value: contributions from |r-r'| < eps_len are 1.

    Returns:
      points_nm : (N,3)
      weights   : (N,)
      W         : (N,3) complex
    """
    pts, wts = build_spherical_quadrature_grid(r0_nm, Nr=Nr, Ntheta=Ntheta, Nphi=Nphi)

    v, k = eval_scalar_mode(pts, r0_nm, l, m, n)
    V = scalar_to_vector(v)

    W = convolve_kernel_collocated(pts, V, wts, eps_len=eps_len, use_gpu=use_gpu)
    return pts, wts, W


def mode_vector_norm_quadrature(points_nm, weights, V):
    """
    Compute integral conj(V).V dV for a given vector field V on a quadrature grid.

    Useful for checking normalization (should be ~1 for calc_mode_field_quadrature output).
    """
    V = np.asarray(V, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    dots = np.einsum('ij,ij->i', np.conj(V), V)
    return np.dot(dots, w)
