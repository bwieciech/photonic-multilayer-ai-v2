"""
Created based on https://github.com/sbyrnes321/tmm from Steven J. Byrnes
https://arxiv.org/abs/1603.02720
"""
import sys
from typing import Optional, Collection, Dict, Union

import numpy as np
import torch

EPSILON = sys.float_info.epsilon  # typical floating-point calc error


def unpolarized_RT_vec(
    n_list, d_list, th_0, wavelengths, device: Optional[Union[str, torch.device]] = None
):
    """
    Vectorized unpolarized RT over:
      - multiple stacks  (n_list.shape=(N,L))
      - multiple angles  (th_0 could be array)
      - multiple lambda  (wavelengths could be array)
    Returns {'R': ..., 'T': ...} with the same shape (N, A, W).
    """
    device = torch.device(device)
    batch_size = 5_000 if device.type == "cuda" else 100
    # We'll just do two TMM calls (s & p) and average.
    s_R = None
    s_T = None
    p_R = None
    p_T = None

    n_rows = n_list.shape[0]
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)

        s_data = coh_tmm_vec(
            "s", n_list[start:end], d_list[start:end], th_0, wavelengths, device
        )
        s_R = s_data["R"] if s_R is None else np.concatenate((s_R, s_data["R"]), axis=0)
        s_T = s_data["T"] if s_T is None else np.concatenate((s_T, s_data["T"]), axis=0)

        p_data = coh_tmm_vec(
            "p", n_list[start:end], d_list[start:end], th_0, wavelengths, device
        )
        p_R = p_data["R"] if p_R is None else np.concatenate((p_R, p_data["R"]), axis=0)
        p_T = p_data["T"] if p_T is None else np.concatenate((p_T, p_data["T"]), axis=0)

    R = 0.5 * (s_R + p_R)
    T = 0.5 * (s_T + p_T)
    return {"R": R, "T": T}


def coh_tmm_vec(
    pol: str,
    n_list,
    d_list,
    th_0,
    wavelengths,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Vectorized TMM over:
      - possibly multiple stacks (n_list.shape=(N,L), d_list.shape=(N,L))
      - possibly multiple angles or multiple wavelengths.
    Returns a dict with arrays of shape (N, W, A) [note: output order is (N, wavelengths, angles)].
    """
    # Convert n_list/d_list into at-least-2D form:
    n_list = np.array(n_list, dtype=object, copy=False)
    d_list = np.array(d_list, dtype=float, copy=False)

    # If the user just gave 1D arrays for n_list & d_list => shape them as (1,L)
    if n_list.ndim == 1:
        n_list = n_list[np.newaxis, :]  # shape => (1,L)
    if d_list.ndim == 1:
        d_list = d_list[np.newaxis, :]  # shape => (1,L)

    th_0 = np.array(th_0, copy=False, ndmin=1)  # shape => (A,)
    wavelengths = np.array(wavelengths, copy=False, ndmin=1)  # shape => (W,)

    N1, L = n_list.shape
    W = wavelengths.size
    A = th_0.size

    # For each element in n_list, compute its refractive indices over wavelengths.
    # (If the element is callable, it is evaluated; otherwise it is broadcast.)
    cache = {}
    calculate_refractive_indices_vec = np.frompyfunc(
        lambda el: calculate_refractive_indices(el, wavelengths, cache), 1, 1
    )
    n_list = np.array(calculate_refractive_indices_vec(n_list).tolist())
    # Expand dims and repeat to add the angle dimension (A)
    # n_list becomes shape (N1, L, W, A)
    n_list = np.expand_dims(n_list, axis=-1).repeat(A, axis=-1)
    # Rearrange axes so that the new shape is (N1, W, A, L) and then flatten over (N1, W, A)
    n_list = n_list.swapaxes(1, 2).swapaxes(2, 3).reshape(N1 * W * A, L)

    N_indices = np.repeat(np.arange(N1), W * A)
    d_list_flat = d_list[N_indices]  # shape: (N1 * W * A, L)
    A_indices = np.tile(np.arange(A), N1 * W)
    W_indices = np.tile(np.repeat(np.arange(W), A), N1)
    th_0_flat = th_0[A_indices]  # shape: (N1 * W * A,)
    wavelengths_flat = wavelengths[W_indices]  # shape: (N1 * W * A,)

    results = coh_tmm_flattened_stacks(
        pol, n_list, d_list_flat, th_0_flat, wavelengths_flat, device
    )

    # Reshape back to (N1, W, A) so that the output shape is the same as before.
    return {
        "R": results["R"].cpu().detach().numpy().reshape((N1, W, A)),
        "T": results["T"].cpu().detach().numpy().reshape((N1, W, A)),
    }


def calculate_refractive_indices(element, wls, cache):
    if callable(element):
        if element not in cache:
            cache[element] = element(wls)
        result = cache[element]
        if np.isscalar(result):  # If the result is a scalar, broadcast it
            return np.full(wls.shape, result)
        return result
    else:
        return np.full(wls.shape, element)  # Broadcast the constant


def make_2x2_array(N, a, b, c, d, device: Optional[Union[str, torch.device]] = None):
    """
    Creates a shape-(N, 2, 2) tensor with blocks [[a, b], [c, d]].
    Similar to np.array([[a, b], [c, d]]), but batched for size N.
    """
    my_array = torch.empty((N, 2, 2), dtype=torch.cdouble, device=device)
    my_array[:, 0, 0] = a
    my_array[:, 0, 1] = b
    my_array[:, 1, 0] = c
    my_array[:, 1, 1] = d
    return my_array


def is_forward_angle(n, theta):
    """
    Decide whether wave is forward-traveling or backward-traveling,
    per https://arxiv.org/abs/1603.02720 Appendix D
    n and theta are 1D (same length).
    """
    if torch.any(n.real * n.imag < -EPSILON):
        raise ValueError(
            "For materials with gain, it's ambiguous which is incoming vs outgoing.\n"
            f"n: {n}, theta: {theta}"
        )
    ncostheta = n * torch.cos(theta)
    return torch.where(
        torch.abs(ncostheta.imag) > 100 * EPSILON,
        ncostheta.imag > 0,
        ncostheta.real > 0,
    )


def list_snell(n_list, th_0):
    """
    Returns the complex angle in each layer, from the incident angle in layer0.
    The first & last angles are forced to be "forward traveling".
    n_list: shape (N, num_layers), each row is one stack
    th_0: scalar or shape (N,)
    returns: angles of shape (N, num_layers)
    """
    sin_in = n_list[:, 0].unsqueeze(-1) * torch.sin(th_0).unsqueeze(-1) / n_list
    angles = torch.asin(sin_in)
    forward0 = is_forward_angle(n_list[:, 0], angles[:, 0])
    angles[:, 0] = torch.where(
        forward0,
        angles[:, 0],
        torch.pi - angles[:, 0],
    )
    forward_last = is_forward_angle(n_list[:, -1], angles[:, -1])
    angles[:, -1] = torch.where(
        forward_last,
        angles[:, -1],
        torch.pi - angles[:, -1],
    )
    return angles


def interface_r(polarization, n_i, n_f, th_i, th_f):
    """
    Fresnel reflection amplitude.
    All inputs are (N,) 1D complex Tensors (same length).
    Returns shape (N,) complex.
    """
    if polarization == "s":
        num = n_i * torch.cos(th_i) - n_f * torch.cos(th_f)
        den = n_i * torch.cos(th_i) + n_f * torch.cos(th_f)
        return num / den
    elif polarization == "p":
        num = n_f * torch.cos(th_i) - n_i * torch.cos(th_f)
        den = n_f * torch.cos(th_i) + n_i * torch.cos(th_f)
        return num / den
    else:
        raise ValueError("polarization must be 's' or 'p'")


def interface_t(polarization, n_i, n_f, th_i, th_f):
    """
    Fresnel transmission amplitude.
    All inputs are (N,) 1D complex Tensors (same length).
    Returns shape (N,) complex.
    """
    if polarization == "s":
        return (
            2 * n_i * torch.cos(th_i) / (n_i * torch.cos(th_i) + n_f * torch.cos(th_f))
        )
    elif polarization == "p":
        return (
            2 * n_i * torch.cos(th_i) / (n_f * torch.cos(th_i) + n_i * torch.cos(th_f))
        )
    else:
        raise ValueError("polarization must be 's' or 'p'")


def R_from_r(r):
    """Reflected power given reflection amplitude r: |r|^2."""
    return torch.abs(r) ** 2


def T_from_t(pol, t, n_i, n_f, th_i, th_f):
    """
    Transmitted power given transmission amplitude t.
    For s-pol:  T = |t|^2 * (n_f cos(th_f)) / (n_i cos(th_i))
    For p-pol:  T = |t|^2 * Re[n_f cos(th_f*)]/Re[n_i cos(th_i*)]
    """
    if pol == "s":
        return (
            (torch.abs(t) ** 2)
            * (n_f * torch.cos(th_f)).real
            / ((n_i * torch.cos(th_i)).real + EPSILON)
        )
    elif pol == "p":
        return (
            (torch.abs(t) ** 2)
            * ((n_f * torch.conj(torch.cos(th_f))).real)
            / ((n_i * torch.conj(torch.cos(th_i))).real + EPSILON)
        )
    else:
        raise ValueError("polarization must be 's' or 'p'")


def power_entering_from_r(pol, r, n_i, th_i):
    """
    Power entering the first interface, which may differ from 1-R if the incident medium is lossy.
    """
    if pol == "s":
        top = (n_i * torch.cos(th_i) * (1 + torch.conj(r)) * (1 - r)).real
        bot = (n_i * torch.cos(th_i)).real + EPSILON
        return top / bot
    elif pol == "p":
        top = (n_i * torch.conj(torch.cos(th_i)) * (1 + r) * (1 - torch.conj(r))).real
        bot = (n_i * torch.conj(torch.cos(th_i))).real + EPSILON
        return top / bot
    else:
        raise ValueError("polarization must be 's' or 'p'")


def coh_tmm_flattened_stacks(
    pol: str,
    n_list: Collection[complex],
    d_list: Collection[float],
    th_0: Collection[float],
    wavelengths: Collection[float],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Main coherent TMM single-stack function, rewritten in PyTorch.
    All returned arrays/tensors are in PyTorch.

    Args:
        pol: 's' or 'p'
        n_list: shape (N, num_layers), complex or real→complex
        d_list: shape (N, num_layers), real
        th_0: scalar (float) or shape (N,) [incident angle in layer0]
        wavelengths: shape (N,) [vacuum wavelength]
        device: torch device
    """
    n_list = torch.tensor(
        n_list, dtype=torch.cdouble, requires_grad=False, device=device
    )
    d_list = torch.tensor(
        d_list, dtype=torch.float64, requires_grad=False, device=device
    )
    wavelengths = torch.tensor(
        wavelengths, dtype=torch.float64, requires_grad=False, device=device
    )
    th_0 = torch.tensor(th_0, dtype=torch.float64, requires_grad=False, device=device)

    if torch.any(~torch.isinf(d_list[:, 0])) or torch.any(~torch.isinf(d_list[:, -1])):
        raise ValueError("d_list must start and end with inf (semi-inf media).")
    test_imag = (n_list[:, 0] * torch.sin(th_0)).imag
    if torch.any(torch.abs(test_imag) > 100 * EPSILON):
        raise ValueError("Check that n0 * sin(th0) is real!")
    forward0 = is_forward_angle(n_list[:, 0], th_0)
    if not torch.all(forward0):
        raise ValueError("Incident angle is not forward traveling?!")

    num_stacks, num_layers = n_list.shape

    # Compute angles in each layer => (N, num_layers)
    th_list = list_snell(n_list, th_0)
    kz_list = 2 * torch.pi * n_list * torch.cos(th_list) / wavelengths[:, None]
    delta = kz_list * d_list

    is_finite = ~torch.isinf(delta.real) & ~torch.isinf(delta.imag)
    big_abs_mask = (delta.imag > 35) & is_finite
    real_part = torch.where(big_abs_mask, delta.real, delta.real)
    imag_part = torch.where(big_abs_mask, 35, delta.imag)
    delta = real_part + 1j * imag_part

    r_list = torch.zeros(
        (num_stacks, num_layers, num_layers), dtype=torch.cdouble, device=device
    )
    t_list = torch.zeros(
        (num_stacks, num_layers, num_layers), dtype=torch.cdouble, device=device
    )
    r_ij = interface_r(
        pol, n_list[:, :-1], n_list[:, 1:], th_list[:, :-1], th_list[:, 1:]
    )
    t_ij = interface_t(
        pol, n_list[:, :-1], n_list[:, 1:], th_list[:, :-1], th_list[:, 1:]
    )
    idx_i = torch.arange(num_layers - 1)
    idx_ip1 = idx_i + 1
    r_list[:, idx_i, idx_ip1] = r_ij
    t_list[:, idx_i, idx_ip1] = t_ij

    M_list = torch.zeros(
        (num_stacks, num_layers, 2, 2), dtype=torch.cdouble, device=device
    )
    for i in range(1, num_layers - 1):
        t_factor = 1.0 / t_list[:, i, i + 1].unsqueeze(-1).unsqueeze(-1)
        exp_matrix = make_2x2_array(
            num_stacks,
            torch.exp(-1j * delta[:, i]),
            torch.zeros_like(delta[:, i]),
            torch.zeros_like(delta[:, i]),
            torch.exp(1j * delta[:, i]),
            device,
        )
        ref_matrix = make_2x2_array(
            num_stacks,
            torch.ones_like(r_list[:, i, i + 1]),
            r_list[:, i, i + 1],
            r_list[:, i, i + 1],
            torch.ones_like(r_list[:, i, i + 1]),
            device,
        )
        M_list[:, i] = t_factor * torch.matmul(exp_matrix, ref_matrix)

    Mtilde = make_2x2_array(
        num_stacks,
        torch.ones(num_stacks, dtype=torch.cdouble),
        r_list[:, 0, 1],
        r_list[:, 0, 1],
        torch.ones(num_stacks, dtype=torch.cdouble),
        device,
    )
    Mtilde = Mtilde / t_list[:, 0, 1].unsqueeze(-1).unsqueeze(-1)

    for i in range(1, num_layers - 1):
        Mtilde = torch.matmul(Mtilde, M_list[:, i])

    r_amp = Mtilde[:, 1, 0] / Mtilde[:, 0, 0]
    t_amp = 1.0 / Mtilde[:, 0, 0]

    vw_list = torch.zeros(
        (num_stacks, num_layers, 2), dtype=torch.cdouble, device=device
    )
    vw = torch.zeros((num_stacks, 2), dtype=torch.cdouble, device=device)
    vw[:, 0] = t_amp
    vw_list[:, -1, :] = vw

    for i in range(num_layers - 2, 0, -1):
        vw_new = torch.matmul(M_list[:, i], vw.unsqueeze(-1))
        vw = vw_new.squeeze(-1)
        vw_list[:, i, :] = vw

    R = R_from_r(r_amp)
    T = T_from_t(pol, t_amp, n_list[:, 0], n_list[:, -1], th_0, th_list[:, -1])
    power_entering = power_entering_from_r(pol, r_amp, n_list[:, 0], th_0)

    return {
        "r": r_amp,
        "t": t_amp,
        "R": R,
        "T": T,
        "power_entering": power_entering,
        "vw_list": vw_list,
        "kz_list": kz_list,
        "th_list": th_list,
        "pol": pol,
        "n_list": n_list,
        "d_list": d_list,
        "th_0": th_0,
        "wavelengths": wavelengths,
    }
