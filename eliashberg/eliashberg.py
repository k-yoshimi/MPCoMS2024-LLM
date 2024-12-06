import numpy as np
import hwave.qlmsio.wan90 as wan90

def get_interaction_matrix_k(kx_array, ky_array, kz_array, value_r, norb):
    """Calculate interaction matrix in k-space from real-space values.

    Parameters
    ----------
    kx_array : ndarray
        Array of kx points in the Brillouin zone
    ky_array : ndarray
        Array of ky points in the Brillouin zone
    kz_array : ndarray
        Array of kz points in the Brillouin zone
    value_r : dict
        Dictionary containing real-space interaction values
        Keys are tuples ((Rx,Ry,Rz), (orb1,orb2))
        Values are interaction strengths
    norb : int
        Number of orbitals

    Returns
    -------
    value_k : ndarray
        5D array of shape (norb,norb,nkx,nky,nkz) containing
        interaction matrix elements in k-space
    """
    value_k = np.zeros((norb, norb, kx_array.size, ky_array.size, kz_array.size), dtype=np.complex64)
    kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx_array, ky_array, kz_array, indexing='ij')
    for key, value in value_r.items():
        #key = (i, j, k, orb1, orb2)
        orb1, orb2 = key[1]
        Rx, Ry, Rz = key[0]
        value_k[orb2, orb1, :, :, :] += value * np.exp(1j * (kx_mesh * Rx + ky_mesh * Ry + kz_mesh * Rz))
    return value_k

def get_green(Nx, Ny, Nz, Nmat, myu, beta, epsilon_k):
    """Calculate the Green's function in Matsubara frequency space.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of k-points in each direction
    Nmat : int
        Number of Matsubara frequencies
    myu : float
        Chemical potential
    beta : float
        Inverse temperature
    epsilon_k : ndarray
        Single-particle energies in k-space

    Returns
    -------
    green_kw : ndarray
        6D array of shape (2,2,Nx,Ny,Nz,Nmat) containing
        the Green's function G(k,iωn)
    
    Notes
    -----
    Calculates G(k,iωn) = Σ_m V_m V_m^† / (iωn - (ε_m - μ))
    where V_m are eigenvectors and ε_m eigenvalues of epsilon_k
    """
    iomega = np.array([(2.0 * i + 1.0 - Nmat) * np.pi for i in range(Nmat)]) / beta
    green_kw = np.zeros((2, 2, Nx, Ny, Nz, Nmat), dtype=np.complex64)
    for idx in range(Nx):
        for idy in range(Ny):
            for idz in range(Nz):
                eigvals, eigvecs = np.linalg.eig(epsilon_k[:, :, idx, idy, idz])
                vec_conj = np.conjugate(eigvecs)
                factor = np.einsum('im,jm->ijm', eigvecs, vec_conj)
                for idmat in range(Nmat):
                    green_kw[:, :, idx, idy, idz, idmat] = np.sum(factor / (1j * iomega[idmat] - (eigvals - myu)[None, None,:]), axis=2)
    return green_kw

def determine_myu(epsilon_k, beta, n):
    """Determine chemical potential using bisection method.

    Parameters
    ----------
    epsilon_k : ndarray
        Single-particle energies in k-space
    beta : float
        Inverse temperature (1/kT)
    n : float
        Target electron filling per orbital

    Returns
    -------
    myu : float
        Chemical potential that gives the desired filling
    """
    from scipy.optimize import bisect
    
    def _calc_n(myu):
        total_n = 0.0
        norb = epsilon_k.shape[0]
        for idx in range(Lx):
            for idy in range(Ly):
                for idz in range(Lz):
                    eigvals, eigvecs = np.linalg.eig(epsilon_k[:, :, idx, idy, idz])
                    eigvals = np.real(eigvals)
                    vec_conj = np.conjugate(eigvecs)
                    
                    x = beta * (eigvals - myu)
                    fermi_factors = np.where(x > 100, 0.0,
                                  np.where(x < -100, 1.0,
                                  1.0 / (1.0 + np.exp(x))))
                    
                    total_n += np.real(np.sum(np.einsum('im,mm,im->i', eigvecs, np.diag(fermi_factors), vec_conj)))
        return float(total_n/(Lx*Ly*Lz) - n * norb)

    myu = bisect(_calc_n, -10.0, 10.0)
    return float(myu)


def calc_chi0q(Nx, Ny, Nz, Nmat, green_a_kw, green_b_kw, myu, beta):
    """Calculate the bare susceptibility χ₀(q,iωn).

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of q-points in each direction
    Nmat : int
        Number of Matsubara frequencies
    green_a_kw, green_b_kw : ndarray
        Green's functions for orbitals a and b
    myu : float
        Chemical potential  
    beta : float
        Inverse temperature

    Returns
    -------
    chi0q : ndarray
        Bare susceptibility χ₀(q,iωn)

    Notes
    -----
    Calculates χ₀ using:
    1. FFT of Green's functions from k to r space
    2. FFT from Matsubara freq to imaginary time
    3. Multiply in r,τ space
    4. FFT back to q,ωn space
    """
    
    from numpy.fft import ifftn
    #q2r FFT
    def _green_q2r_fft(green_kw):
        green_rw = ifftn(green_kw, axes=(0, 1, 2))
        return green_rw

    green_a_rw = _green_q2r_fft(green_a_kw)
    green_b_rw = _green_q2r_fft(green_b_kw)
        
    from numpy.fft import fft
    #mat2tau FFT
    def _green_mat2tau_fft(green_rw, Nmat):
        green_rt = fft(green_rw, axis=3)
        delta_omega = 1.0 / Nmat - 1.0
        idmat = np.arange(Nmat)
        exp_factor = np.exp(-1J * np.pi * idmat * delta_omega)
        green_rt *= exp_factor
        return green_rt

    green_a_rt = _green_mat2tau_fft(green_a_rw, Nmat)
    green_b_rt = _green_mat2tau_fft(green_b_rw, Nmat)
    
    #Calculate Chi(r, t)
    idx = np.arange(Nx)
    idy = np.arange(Ny)
    idz = np.arange(Nz)
    idmat = np.arange(Nmat)

    idx_inv = (Nx - idx) % Nx
    idy_inv = (Ny - idy) % Ny
    idz_inv = (Nz - idz) % Nz
    idmat_inv = (Nmat - idmat - 1) % Nmat

    green_b_rt_inv = green_b_rt[idx_inv[:, None, None, None], idy_inv[None, :, None, None], idz_inv[None, None, :, None], idmat_inv[None, None, None, :]]
    chi_rt = green_a_rt * green_b_rt_inv
    chi_rt[:, :, :, 1:] *= -1  # Apply the condition for idmat != 0

    #r2q FFT
    from numpy.fft import fftn, ifft
    chi_qt = fftn(chi_rt, axes=(0, 1, 2))

    delta_omega = -1.0
    idmat = np.arange(Nmat)
    exp_factor = np.exp(1J * np.pi * idmat * delta_omega)

    chi_qt *= exp_factor  # Apply the exponential factor
    chi_qw = ifft(chi_qt, axis=3)  # Apply ifft along the third axis (tau -> mat)
    return -chi_qw/beta

import numba as nb
@nb.jit(nopython=True, fastmath=True)
def compute_inverse(mat):
    return np.linalg.inv(mat)
    #return scipy.linalg.solve(mat, I, assume_a='her')

@nb.jit(nopython=True, fastmath=True)
def compute_matrices(Pc_q, Ps_q, chi0q, Vq, U, norb, Lx, Ly, Lz, Nmat):
    """Calculate charge and spin vertices using RPA.

    Parameters
    ----------
    Pc_q, Ps_q : ndarray
        Output arrays for charge/spin vertices
    chi0q : ndarray
        Bare susceptibility
    Vq : ndarray
        Non-local Coulomb interaction
    U : ndarray
        Local Coulomb interaction
    norb : int
        Number of orbitals
    Lx, Ly, Lz : int
        Number of q-points
    Nmat : int
        Number of Matsubara frequencies

    Notes
    -----
    Calculates RPA vertices:
    Pc = (Wc + Ws)/2 - WcχcWc/2
    Ps = -Ws + 3WsχsWs/2
    where Wc = U + 2V, Ws = -U
    """
    I = np.identity(norb)
    chi0q = chi0q.astype(np.complex128)
    Ws = np.empty_like(I, dtype=np.complex128)
    Wc = np.empty_like(I, dtype=np.complex128)
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                _Vq = Vq[:, :, i, j, k]
                _U = U[:, :, i, j, k]
                Wc[:] = _U + 2.0 * _Vq
                Ws[:] = -_U
                _chi0q = np.ascontiguousarray(chi0q[:, :, i, j, k, Nmat//2])
                chis_mat = compute_inverse(I + np.dot(_chi0q, Ws)) @ _chi0q
                chic_mat = compute_inverse(I + np.dot(_chi0q, Wc)) @ _chi0q
                WsChisWs = Ws @ chis_mat @ Ws
                WcChicWc = Wc @ chic_mat @ Wc
                Pc_q[:,:,i,j,k] = (Wc + Ws) / 2.0  - 1.0 / 2.0 * WcChicWc
                Ps_q[:,:,i,j,k] = - Ws + 3.0 / 2.0 * WsChisWs

def calc_sigma_direct_oneshot(green_kw, P, sigma_old):
    """Calculate self-energy using direct convolution method.

    Parameters
    ----------
    green_kw : ndarray
        Single-particle Green's function
    P : ndarray
        Interaction vertex (Pc + Ps)
    sigma_old : ndarray
        Previous iteration self-energy for self-consistent calc

    Returns
    -------
    sigma : ndarray
        Updated self-energy Σ(k)

    Notes
    -----
    Calculates Σ(k) = -(1/N)Σ_q P(q)G₂(k,q)Σ_old(k-q)
    using direct convolution in k-space
    """
    norb = green_kw.shape[0]
    Nx = green_kw.shape[2]
    Ny = green_kw.shape[3]
    Nz = green_kw.shape[4]
    sigma = np.zeros_like(P)
    G2Sigma = np.zeros((norb, norb, Nx, Ny, Nz), dtype=np.complex64)
    for i in range(Nx):
        i_inv = (Nx - i) % Nx
        for j in range(Ny):
            j_inv = (Ny - j) % Ny
            for k in range(Nz):
                k_inv = (Nz - k) % Nz
                gk_w = green_kw[:, :, i, j, k, :]
                gk_w_inv = green_kw[:, :, i_inv, j_inv, k_inv, ::-1]
                G2Sigma[:, :, i, j, k] = np.einsum("ijk, lmk, jm -> il", gk_w, gk_w_inv, sigma_old[:, :, i, j, k])

    for i0 in range(Nx):
        for j0 in range(Ny):
            for k0 in range(Nz):
                _sigma = np.zeros_like(sigma[:,:,i0,j0,k0])
                for i in range(Nx):
                    for j in range(Ny):
                        for k in range(Nz):
                            P_ = P[:, :, (i0 - i) % Nx, (j0 - j) % Ny, (k0 - k) % Nz]
                            _sigma += np.einsum('ij, ij -> ij', P_, G2Sigma[:,:,i,j,k])
                sigma[:, :, i0, j0, k0] = -_sigma/(Nx*Ny*Nz)
    return sigma

def calc_g2(green_kw):
    """Calculate two-particle Green's function G₂.

    Parameters
    ----------
    green_kw : ndarray
        Single-particle Green's function G(k,ω)

    Returns
    -------
    G2 : ndarray
        Two-particle Green's function G₂(k,k',q)
        Shape: (norb,norb,norb,norb,Nx,Ny,Nz)

    Notes
    -----
    Calculates G₂ = G(k)G(-k+q) using FFT convolution
    """
    green_kw_inv = np.roll(green_kw[:, :, ::-1, ::-1, ::-1, ::-1],(1,1,1), (2,3,4))
    G2 = np.einsum("ijpqsk, lmpqsk -> ijlmpqs", green_kw, green_kw_inv)
    return G2

def calc_sigma_fft_one_shot(green_kw, P, G2, sigma_old):
    """Calculate self-energy using FFT convolution method.

    Parameters
    ----------
    green_kw : ndarray
        Single-particle Green's function
    P : ndarray
        Interaction vertex (Pc + Ps)
    G2 : ndarray
        Two-particle Green's function
    sigma_old : ndarray
        Previous iteration self-energy

    Returns
    -------
    sigma : ndarray
        Updated self-energy Σ(k)

    Notes
    -----
    Calculates Σ(k) using FFT convolution:
    1. Transform P(q) and G₂(k,q)Σ(k-q) to real space
    2. Multiply in real space
    3. Transform back to k-space
    More efficient than direct convolution for large system sizes
    """
    norb = green_kw.shape[0]
    G2Sigma = np.einsum("ijlmpqs, jmpqs -> ilpqs", G2, sigma_old)

    #r2q FFT
    from numpy.fft import fftn, ifftn
    def _green_q2r_fft(green_kw):
        green_rw = ifftn(green_kw, axes=(0, 1, 2))
        return green_rw

    P_r_vec = np.array([[_green_q2r_fft(P[i, j, :, :, :]) for j in range(norb)] for i in range(norb)])
    G2Sigma_r_vec = np.array([[_green_q2r_fft(G2Sigma[i, j, :, :, :]) for j in range(norb)] for i in range(norb)])
    Sigma_r_vec = P_r_vec * G2Sigma_r_vec
    print(Sigma_r_vec.shape)
    sigma = fftn(Sigma_r_vec, axes=(-3, -2, -1))
    return -sigma

def initialize_sigma(norb, kx_array, ky_array, kz_array):
    """Initialize the self-energy with a cosine function.
    
    Parameters
    ----------
    norb : int
        Number of orbitals
    kx_array, ky_array, kz_array : ndarray
        k-space grid points
        
    Returns
    -------
    sigma_init : ndarray
        Initialized self-energy (shape: norb, norb, Nx, Ny, Nz)
        Initialized with cos(kx + ky + kz) and normalized
    """
    I = np.identity(norb)
    # Create mesh grid
    kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx_array, ky_array, kz_array, indexing='ij')
    # Calculate cosine term
    cos_term = np.cos(kx_mesh + ky_mesh + kz_mesh)
    # Use broadcasting to compute initial sigma
    sigma_init = I[:, :, None, None, None] * cos_term[None, None, :, :, :]
    return sigma_init / np.linalg.norm(sigma_init)

def save_sigma_to_file(sigma, kx_array, ky_array, kz_array, filename="sigma.dat"):
    """Save self-energy data to file in k-space.
    
    Parameters
    ----------
    sigma : ndarray
        Self-energy array of shape (norb, norb, Nx, Ny, Nz)
    kx_array, ky_array, kz_array : ndarray
        k-space grid points
    filename : str, optional
        Output filename (default: "sigma.dat")
        
    Notes
    -----
    File format: kx ky kz Re(Σ₀₀) Re(Σ₀₁)
    k-coordinates are shifted to [-π, π] range
    """
    Lx, Ly, Lz = len(kx_array), len(ky_array), len(kz_array)
    
    with open(filename, "w") as fw:
        for idx in range(Lx):
            kx = -2.0*np.pi+kx_array[idx] if kx_array[idx] > np.pi else kx_array[idx]
            for idy in range(Ly):
                ky = -2.0*np.pi+ky_array[idy] if ky_array[idy] > np.pi else ky_array[idy]
                for idz in range(Lz):
                    kz = -2.0*np.pi+kz_array[idz] if kz_array[idz] > np.pi else kz_array[idz]
                    fw.write("{} {} {} {} {}\n".format(kx, ky, kz,
                            sigma[0, 0, idx, idy, idz].real,
                            sigma[0, 1, idx, idy, idz].real))

def plot_sigma_3d(sigma, Lx, Ly):
    """Create 3D surface plots of the real part of self-energy.
    
    Parameters
    ----------
    sigma : ndarray
        Self-energy array of shape (norb, norb, Nx, Ny, Nz)
    Lx, Ly : int
        Number of k-points in x and y directions
    
    Notes
    -----
    Creates two 3D plots:
    - Left: Real part of Σ₀₀(k) (diagonal component)
    - Right: Real part of Σ₀₁(k) (off-diagonal component)
    Both plotted in the kx-ky plane at kz = 0
    """
    import matplotlib.pyplot as plt
    
    # Shift k-coordinates to [-π, π] range
    x_shifted = np.linspace(-np.pi, np.pi, Lx)
    y_shifted = np.linspace(-np.pi, np.pi, Ly)
    Y_shifted, X_shifted = np.meshgrid(x_shifted, y_shifted)

    # Extract and shift real parts of self-energy components
    Z1 = np.real(sigma[0, 0, :, :, 0])  # Diagonal component Σ₀₀
    Z1_shifted = np.fft.fftshift(Z1)
    Z2 = np.real(sigma[0, 1, :, :, 0])  # Off-diagonal component Σ₀₁
    Z2_shifted = np.fft.fftshift(Z2)

    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(14, 6))

    # Plot diagonal component
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_shifted, Y_shifted, Z1_shifted, cmap='viridis')
    ax1.set_title("norb1 = 0, norb2 = 0")
    ax1.set_xlabel("kx")
    ax1.set_ylabel("ky")
    ax1.set_zlabel("Real(sigma)")

    # Plot off-diagonal component
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_shifted, Y_shifted, Z2_shifted, cmap='viridis')
    ax2.set_title("norb1 = 0, norb2 = 1")
    ax2.set_xlabel("kx")
    ax2.set_ylabel("ky")
    ax2.set_zlabel("Real(sigma)")

    plt.tight_layout()
    plt.show()

def load_or_calculate_green(calc_flag, kx_array, ky_array, kz_array, hr, norb, 
                          Lx, Ly, Lz, Nmat, beta, n_filling):
    """Calculate or load Green's function from file.

    Parameters
    ----------
    calc_flag : bool
        If True, calculate from scratch. If False, load from file
    kx_array, ky_array, kz_array : ndarray
        k-space grid points
    hr : dict
        Hopping parameters
    norb : int
        Number of orbitals
    Lx, Ly, Lz : int
        System size in each direction
    Nmat : int
        Number of Matsubara frequencies
    beta : float
        Inverse temperature
    n_filling : float
        Target electron filling

    Returns
    -------
    green_kw : ndarray
        Green's function
    myu : float
        Chemical potential (valid only when calculating from scratch)
    """
    filename = f"g0q_beta{beta}_Lx{Lx}_Ly{Ly}_Lz{Lz}_Nmat{Nmat}.npy"
    
    if calc_flag:
        print("Calculate epsilon_k")
        epsilon_k = get_interaction_matrix_k(kx_array, ky_array, kz_array, hr, norb)
        myu = determine_myu(epsilon_k, beta, n_filling)
        print("Calculate green_kw")
        green_kw = get_green(Lx, Ly, Lz, Nmat, myu, beta, epsilon_k)
        np.save(filename, green_kw)
    else:
        print("Load green_kw")
        green_kw = np.load(filename)
        myu = None  # Not needed when loading from file
        
    return green_kw, myu

def load_or_calculate_chi0q(calc_flag, green_kw, norb, Lx, Ly, Lz, Nmat, myu, beta):
    """Calculate or load susceptibility from file.

    Parameters
    ----------
    calc_flag : bool
        If True, calculate from scratch. If False, load from file
    green_kw : ndarray
        Green's function
    norb : int
        Number of orbitals
    Lx, Ly, Lz : int
        System size in each direction
    Nmat : int
        Number of Matsubara frequencies
    myu : float
        Chemical potential
    beta : float
        Inverse temperature

    Returns
    -------
    chi0q : ndarray
        Bare susceptibility
    """
    filename = f"chi0q_beta{beta}_Lx{Lx}_Ly{Ly}_Lz{Lz}_Nmat{Nmat}.npy"
    
    if calc_flag:
        print("Calculate chi0q")
        chi0q = np.zeros((norb, norb, Lx, Ly, Lz, Nmat), dtype=np.complex64)
        for iorb in range(norb):
            for jorb in range(norb):
                chi0q[iorb][jorb] = calc_chi0q(Lx, Ly, Lz, Nmat, 
                                             green_kw[iorb][jorb], 
                                             green_kw[jorb][iorb], 
                                             myu, beta)[:,:,:]
        np.save(filename, chi0q)
    else:
        print("Load chi0q")
        chi0q = np.load(filename)
        
    return chi0q

def solve_self_consistent_sigma(green_kw, Pc_q, Ps_q, G2, sigma_init, 
                              max_iter=1000, alpha=0.5, tol=1e-5):
    """Solve self-consistent equation for self-energy.

    Parameters
    ----------
    green_kw : ndarray
        Green's function
    Pc_q, Ps_q : ndarray
        Charge and spin vertices
    G2 : ndarray
        Two-particle Green's function
    sigma_init : ndarray
        Initial guess for self-energy
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    alpha : float, optional
        Mixing parameter for convergence (default: 0.5)
    tol : float, optional
        Convergence tolerance (default: 1e-5)

    Returns
    -------
    sigma : ndarray
        Converged self-energy
    converged : bool
        Whether the calculation converged
    n_iter : int
        Number of iterations performed
    """
    sigma_old = sigma_init.copy()
    
    for iteration in range(max_iter):
        # Calculate new self-energy
        sigma = calc_sigma_fft_one_shot(green_kw, Pc_q + Ps_q, G2, sigma_old)
        norm = np.linalg.norm(sigma)
        
        # Check convergence
        diff = np.linalg.norm(sigma/norm - sigma_old)
        print(f"Iteration: {iteration}, norm: {norm:.6f}, diff: {diff:.6f}")
        
        if diff < tol:
            print("Converged.")
            sigma = sigma/norm
            return sigma, True, iteration + 1
            
        # Mix old and new solutions
        sigma_old = (1.0 - alpha) * sigma.copy()/norm + alpha * sigma_old
        
    print(f"Failed to converge after {max_iter} iterations.")
    return sigma_old, False, max_iter

if __name__ == "__main__":
    """Main execution script for Eliashberg equation solver.
    
    This script performs the following steps:
    1. Initialize system parameters
    2. Calculate/load non-interacting Green's function
    3. Calculate/load bare susceptibility
    4. Compute RPA vertices
    5. Solve self-consistent equation for self-energy
    6. Save and visualize results
    
    System Parameters
    ----------------
    Lx, Ly, Lz : int
        System size in each direction (32, 32, 1)
    Nmat : int
        Number of Matsubara frequencies (512)
    beta : float
        Inverse temperature (10.0)
    n_filling : float
        Target electron filling per orbital (0.75)
    
    Input Files
    -----------
    geom.dat : file
        Contains geometry information and number of orbitals
    Transfer.dat : file
        Contains hopping parameters in real space
    coulombinter.dat : file
        Contains non-local Coulomb interaction
    coulombintra.dat : file
        Contains local Coulomb interaction
    
    Output Files
    -----------
    g0q_beta{}_Lx{}_Ly{}_Lz{}_Nmat{}.npy : file
        Saved non-interacting Green's function
    chi0q_beta{}_Lx{}_Ly{}_Lz{}_Nmat{}.npy : file
        Saved bare susceptibility
    sigma.dat : file
        Final self-energy in k-space
    """
    
    # System parameters
    Lx, Ly, Lz = 32, 32, 1
    Nmat = 512
    calc_g0q_flag = True    # Calculate Green's function from scratch if True
    calc_chi0q_flag = True  # Calculate susceptibility from scratch if True
    beta = 10.0            # Inverse temperature
    n_filling = 0.75       # Target electron filling
    
    # Read input files
    geom_info = wan90.read_geom("geom.dat")
    norb = geom_info["norb"]
    hr = wan90.read_w90("Transfer.dat")        # Hopping parameters
    vr = wan90.read_w90("coulombinter.dat")    # Non-local Coulomb
    ur = wan90.read_w90("coulombintra.dat")    # Local Coulomb
    
    # Setup k-space mesh
    kx_array = np.linspace(0, 2.*np.pi, Lx, endpoint=False)
    ky_array = np.linspace(0, 2.*np.pi, Ly, endpoint=False)
    kz_array = np.linspace(0, 2.*np.pi, Lz, endpoint=False)
    
    # Calculate or load Green's function
    green_kw, myu = load_or_calculate_green(calc_g0q_flag, kx_array, ky_array, kz_array, hr, norb, 
                                          Lx, Ly, Lz, Nmat, beta, n_filling)
    
    # Calculate or load susceptibility
    chi0q = load_or_calculate_chi0q(calc_chi0q_flag, green_kw, norb, Lx, Ly, Lz, Nmat, myu, beta)
    
    # Initialize RPA vertices
    chicq = np.zeros((kx_array.size, ky_array.size, kz_array.size, norb), dtype=np.complex128)
    chisq = np.zeros_like(chicq)
    Pc_q = np.zeros((norb, norb, kx_array.size, ky_array.size, kz_array.size), dtype=np.complex128)
    Ps_q = np.zeros_like(Pc_q)
    
    # Calculate interaction vertices in k-space
    Vq = get_interaction_matrix_k(kx_array, ky_array, kz_array, vr, norb)
    U = get_interaction_matrix_k(kx_array, ky_array, kz_array, ur, norb)
    compute_matrices(Pc_q, Ps_q, chi0q, Vq, U, norb, Lx, Ly, Lz, Nmat)
    
    # Initialize self-energy with cosine function
    sigma_init = initialize_sigma(norb, kx_array, ky_array, kz_array)
    
    # Calculate G2 for sigma calculation
    G2 = calc_g2(green_kw)
    
    # Self-consistent iteration for self-energy
    sigma, converged, n_iter = solve_self_consistent_sigma(
        green_kw=green_kw, 
        Pc_q=Pc_q, 
        Ps_q=Ps_q, 
        G2=G2, 
        sigma_init=sigma_init
    )
    
    # Save self-energy to file
    save_sigma_to_file(sigma, kx_array, ky_array, kz_array)
    # Plot self-energy
    plot_sigma_3d(sigma, Lx, Ly)

