import einops
import torch
import math
import pdb
import os

def embedding_truncation(
    X: torch.Tensor, retain_k=2,
) -> torch.Tensor:
    orig_shape = X.shape

    if len(X.shape) == 3:
        B, N, D = X.shape
        X_new = torch.zeros_like(X)

        for b in range(B):
            U, S, Vh = torch.linalg.svd(X[b], full_matrices=False)
          
            energy_topk = torch.sum(S[:retain_k] ** 2)
            energy_total = torch.sum(S ** 2)
            # print(f"[Top-k = {retain_k}] Kept Energy Ratio:", energy_topk / energy_total)
            
            S_trunc = S.clone()
            S_trunc[retain_k:] = 0  # truncation

            X_new[b] = U @ torch.diag(S_trunc) @ Vh

    elif len(X.shape) == 2:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        S_trunc = S.clone()
        S_trunc[retain_k:] = 0

        X_new = U @ torch.diag(S_trunc) @ Vh

    else:
        raise ValueError(f"Unexpected input shape {X.shape}, expected (B, N, D) or (N, D).")

    return X_new.to(torch.float16)

def embedding_svd_topk_exponential(
    X: torch.Tensor, 
    top_k: int = 8, 
    alpha: float = 0.01, 
    beta: float = 1.0, 
) -> torch.Tensor:
    orig_shape = X.shape
    S_before_all, S_after_all = [], []

    if X.dim() == 3:
        B, N, D = X.shape
        X_out = torch.zeros_like(X)
        
        for b in range(B):
            U, S, Vh = torch.linalg.svd(X[b], full_matrices=False)  # S: (min(N,D),)

            S_new = S.clone()
            # print(f"[INFO] before {S}")
            if top_k < len(S):
                tail = S[top_k:]
                S_new[top_k:] = beta * torch.exp(-alpha * tail) * tail

            # print(f"[INFO] after {S_new}")
            X_out[b] = U @ torch.diag(S_new) @ Vh

    elif X.dim() == 2:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S_new = S.clone()
        if top_k < len(S):
            tail = S[top_k:]
            S_new[top_k:] = beta * torch.exp(-alpha * tail) * tail
        X_out = U @ torch.diag(S_new) @ Vh
    else:
        raise ValueError(f"Input shape {X.shape} not supported. Must be (B, N, D) or (N, D).")

    return X_out.to(torch.float16)

def embedding_svd_topk_exponential_timeaware(
    X: torch.Tensor, 
    top_k: int = 8, 
    alpha: float = 0.1, 
    beta: float = 1.0,
    step: int = 0,
    total_steps: int = 50,
    gamma: float = 10.0,
    center: float = 0.3,
) -> torch.Tensor:
    t = step/total_steps
    s_t = 1 / (1 + math.exp(-gamma * (t - center)))
    alpha_t = alpha * (1-s_t)
    
    orig_shape = X.shape

    if X.dim() == 3:
        B, N, D = X.shape
        X_out = torch.zeros_like(X)

        for b in range(B):
            U, S, Vh = torch.linalg.svd(X[b], full_matrices=False)

            S_new = S.clone()
            if top_k < len(S):
                tail = S[top_k:]
                S_new[top_k:] = beta * torch.exp(-alpha_t * tail) * tail

            # print(f"{S} vs. {S_new}")
            X_out[b] = U @ torch.diag(S_new) @ Vh

    elif X.dim() == 2:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S_new = S.clone()
        if top_k < len(S):
            tail = S[top_k:]
            S_new[top_k:] = beta * torch.exp(-alpha_t * tail) * tail

        X_out = U @ torch.diag(S_new) @ Vh

    else:
        raise ValueError(f"Input shape {X.shape} not supported. Must be (B, N, D) or (N, D).")

    return X_out.to(torch.float16)
