import torch
from typing import Tuple
import torch.nn.functional as F

def safe_kl_pd_pc(log_pd: torch.Tensor, log_pc: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(log_pd) & torch.isfinite(log_pc)
    p_d = torch.exp(torch.where(finite, log_pd, torch.tensor(-torch.inf, device=log_pd.device)))  # exp(-inf)=0
    diff = torch.where(finite, log_pd - log_pc, torch.zeros_like(log_pd))

    kl = (p_d * diff).sum(dim=-1)
    return kl.clamp_min(0.0)

def solve_optimization_newton(
        clean_logits: torch.Tensor,
        dirty_logits: torch.Tensor,
        k_radius,  # float or Tensor [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bc, bd = _solve_theta_newton(clean_logits, dirty_logits, k_radius)
        # If bc/bd are bf16, abs(...) < 1e-6 can fail due to precision; use fp32 in the check.
        assert torch.allclose((bc + bd).float(), torch.ones_like((bc + bd).float()), atol=1e-5, rtol=0.0)
        return bc, bd

@torch.no_grad()
def _solve_theta_newton(
    clean_logits: torch.Tensor,   # [B,V]
    dirty_logits: torch.Tensor,   # [B,V]
    k_radius,                     # float or Tensor [B]
    max_iter: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve for theta = w_d in q_theta ∝ p_c^{1-theta} p_d^{theta} s.t. KL(q_theta || p_c) <= k_t.

    Guarantee-critical properties:
    - All computations in fp32.
    - Robust to masked tokens (-inf) and -inf - -inf -> NaN.
    - Returns a theta that is *numerically feasible* via a final bisection projection.
    """
    device = clean_logits.device
    B, V = clean_logits.shape

    # fp32 log-probs
    log_pd = F.log_softmax(dirty_logits.float(), dim=-1)  # [B,V] fp32
    log_pc = F.log_softmax(clean_logits.float(), dim=-1)  # [B,V] fp32

    # k_t as fp32 [B]
    k_t = torch.as_tensor(k_radius, device=device, dtype=torch.float32)
    if k_t.ndim == 0:
        k_t = k_t.expand(B)
    else:
        k_t = k_t.view(-1)
        assert k_t.numel() == B, f"k_t must be scalar or shape [B], got {k_t.shape}"

    # Corners
    mask_force_pc = (k_t <= 0.0)

    # KL(p_d || p_c) robust (fp32)
    KL_pd_pc = safe_kl_pd_pc(log_pd, log_pc)  # [B]
    mask_use_pd = (KL_pd_pc <= k_t) & (~mask_force_pc)

    active = ~(mask_force_pc | mask_use_pd)

    # Output weights (fp32)
    w_c = torch.empty((B, 1), device=device, dtype=torch.float32)
    w_d = torch.empty((B, 1), device=device, dtype=torch.float32)

    w_c[mask_force_pc] = 1.0
    w_d[mask_force_pc] = 0.0
    w_c[mask_use_pd] = 0.0
    w_d[mask_use_pd] = 1.0

    if not active.any():
        return w_c, w_d  # fp32

    # Active subset
    log_pc_a = log_pc[active]   # [Ba,V] fp32
    log_pd_a = log_pd[active]   # [Ba,V] fp32
    k_a = k_t[active]           # [Ba] fp32
    Ba = log_pc_a.size(0)

    # a = log p_d - log p_c ; sanitize to avoid NaNs/Infs from masking
    a = log_pd_a - log_pc_a
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)  # [Ba,V]

    # Helper: robust KL(q_theta || p_c)
    def kl_theta(th: torch.Tensor) -> torch.Tensor:
        q_log_unnorm = log_pc_a + th[:, None] * a
        logZ = torch.logsumexp(q_log_unnorm, dim=-1)          # [Ba]
        log_q = q_log_unnorm - logZ[:, None]                  # [Ba,V]
        return safe_kl_pd_pc(log_q, log_pc_a)           # [Ba]

    # Bracket in theta-space
    lo = torch.zeros(Ba, device=device, dtype=torch.float32)
    hi = torch.ones(Ba, device=device, dtype=torch.float32)

    # Initialize theta away from endpoints
    theta = torch.clamp(k_a / (k_a + 1.0), 1e-4, 1.0 - 1e-4)

    eps = 1e-8

    for _ in range(max_iter):
        # log q_theta ∝ log_pc + theta*a
        q_log_unnorm = log_pc_a + theta[:, None] * a
        logZ = torch.logsumexp(q_log_unnorm, dim=-1)          # [Ba]
        log_q = q_log_unnorm - logZ[:, None]                  # [Ba,V]
        q = log_q.exp()                                       # [Ba,V]

        # Moments under q (fp32)
        mean_a = (q * a).sum(dim=-1)                          # [Ba]
        mean_a2 = (q * (a * a)).sum(dim=-1)                   # [Ba]
        var_a = (mean_a2 - mean_a * mean_a).clamp_min(0.0)    # [Ba]

        # KL(q||pc) = theta*E[a] - logZ
        KL = theta * mean_a - logZ                            # [Ba]
        # If KL has NaNs (shouldn't, but just in case), treat as infeasible
        KL = torch.nan_to_num(KL, nan=float("inf"), posinf=float("inf"), neginf=0.0)

        f = KL - k_a                                          # [Ba]

        # Update bracket: f<=0 is feasible
        hi = torch.where(f > 0, theta, hi)
        lo = torch.where(f <= 0, theta, lo)

        # Newton step: f' = theta * Var_q[a]
        fp = (theta * var_a).clamp_min(eps)
        theta_new = theta - f / fp

        # Safeguard: stay inside (lo, hi), else bisect
        bad = (theta_new <= lo) | (theta_new >= hi) | ~torch.isfinite(theta_new)
        theta = torch.where(bad, 0.5 * (lo + hi), theta_new)

        if (hi - lo).max() < 1e-6:
            break

    # --- Final feasibility projection (guarantee-critical) ---
    # Ensure returned theta is numerically feasible under kl_theta
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        KL_mid = kl_theta(mid)
        feas = (KL_mid <= k_a)
        lo = torch.where(feas, mid, lo)
        hi = torch.where(feas, hi, mid)

    theta = lo  # feasible by construction (KL <= k_a)

    wd = theta[:, None]              # [Ba,1]
    wc = 1.0 - wd

    w_c[active] = wc
    w_d[active] = wd

    return w_c, w_d  # fp32
