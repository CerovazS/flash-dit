"""Triton kernels for Muon's Newton-Schulz inner loop.

Vendored from https://github.com/KellerJordan/modded-nanogpt (MIT License).
Authors of the original kernels:
- ``XXT`` / ``XTX``: @byronxu99
- ``ba_plus_cAA``: @byronxu99

All three exploit the symmetry of the output matrix to skip the lower-triangle
blocks and mirror the computed upper triangle — saving ~50% of the FLOPs vs
a dense matmul on the same tensor. They run on sm_80 (A100) and sm_86 (3090)
and above; block-size configs are tuned for H100 but work correctly (if not
optimally) on Ampere.

Usage pattern inside a Newton-Schulz iteration:

    # wide case (R <= C)
    XXT(X, out=A)                                  # A = X @ X.T
    ba_plus_cAA(A, alpha=c, beta=b, out=B)         # B = c*A@A + b*A
    torch.addmm(X, B, X, beta=a, out=X_next)       # X_next = a*X + B @ X

    # tall case (R > C), symmetric alternative
    XTX(X, out=A)                                  # A = X.T @ X
    ba_plus_cAA(A, alpha=c, beta=b, out=B)
    torch.addmm(X, X, B, beta=a, out=X_next)       # X_next = a*X + X @ B
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _pick_block_config(K: int) -> tuple[int, int, int, int, int]:
    """Choose (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps).

    The upstream modded-nanogpt configs are tuned for H100 (sm_90) and exceed
    the 101 KB/SM shared-memory budget on Ampere (sm_80 A100, sm_86 RTX 3090).
    For non-Hopper devices we drop ``num_stages`` to 2 which keeps the same
    block sizes running within the shared-memory envelope while keeping the
    same register + warp budget.
    """
    is_hopper = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 9
    )
    if K == 768:
        BM, BN, BK = 128, 128, 64
    else:
        BM, BN, BK = 64, 128, 128
    num_stages = 4 if is_hopper else 2
    num_warps = 8
    return BM, BN, BK, num_stages, num_warps


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


# ---------------------------------------------------------------------------
# XXT: C = A @ A.T (symmetric output)
# ---------------------------------------------------------------------------


@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XXT(A: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute ``C = A @ A.T`` with symmetric-output optimisation.

    ``A`` is `(M, K)` or `(B, M, K)`; ``out`` has shape `(M, M)` or `(B, M, M)`.
    """
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert out.size(-2) == M and out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = _pick_block_config(K)

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    XXT_kernel[grid](
        A_ptr=A, C_ptr=out, M=M, K=K,
        a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_stages=num_stages, num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# XTX: C = A.T @ A (symmetric output)
# ---------------------------------------------------------------------------


@triton.jit
def XTX_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, k_idx, n_idx = _pid_to_block(
        pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_k = (k_idx + tl.arange(0, BLOCK_SIZE_M)) % K
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % K
    offs_m = tl.arange(0, BLOCK_SIZE_K)

    at_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_remaining = M - m * BLOCK_SIZE_K
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        accumulator = tl.dot(at.T, a, accumulator)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs += BLOCK_SIZE_K * a_stride_r

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < K)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < K) & (offs_ck[None, :] < K)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XTX(A: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute ``C = A.T @ A`` with symmetric-output optimisation.

    ``A`` is `(M, K)` or `(B, M, K)`; ``out`` has shape `(K, K)` or `(B, K, K)`.
    Preferred over ``XXT`` when ``M > K`` (tall matrices) since the output is
    small (K×K) instead of (M×M).
    """
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert out.size(-2) == K and out.size(-1) == K

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = _pick_block_config(K)

    grid = (batch_size * triton.cdiv(K, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)
    XTX_kernel[grid](
        A_ptr=A, C_ptr=out, M=M, K=K,
        a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_stages=num_stages, num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# ba_plus_cAA: C = alpha * (A @ A) + beta * A  (square symmetric A)
# ---------------------------------------------------------------------------


@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_remaining = M - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ba_plus_cAA(
    A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor,
) -> torch.Tensor:
    """Compute ``C = alpha * (A @ A) + beta * A`` for a *square, symmetric* ``A``.

    ``A`` can be `(M, M)` or `(B, M, M)`. The square constraint comes from the
    kernel treating ``A`` as its own transpose in the symmetric accumulator.
    """
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert M == K, "Input must be square (symmetric Gram matrix)"
    assert out.size(-2) == M and out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # ba_plus_cAA always operates on a square M×M Gram matrix; use K=M for config selection.
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = _pick_block_config(M)

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    ba_plus_cAA_kernel[grid](
        A_ptr=A, C_ptr=out, M=M,
        a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        alpha=alpha, beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_stages=num_stages, num_warps=num_warps,
    )
    return out
