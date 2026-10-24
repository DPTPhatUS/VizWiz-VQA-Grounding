"""Constrain actual Adam/AdamW displacements, not preconditioned raw gradients."""

from itertools import combinations

import torch
import torch.distributed as dist

from training.size_objective import validate_group_weights


@torch.no_grad()
def parameter_vector(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


@torch.no_grad()
def copy_parameters(parameters, vector):
    position = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.copy_(vector[position : position + count].reshape_as(parameter))
        position += count


def gradient_vector(parameters, scale=1.0):
    if any(p.grad is None for p in parameters):
        raise RuntimeError(
            "Missing gradient in group backward; do not hide unused trainable parameters"
        )
    return torch.cat([p.grad.detach().float().flatten() for p in parameters]) / scale


@torch.no_grad()
def normalized_gradients(gradients):
    rows = gradients.double()
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-20)


@torch.no_grad()
def project_displacement(proposal, gradients):
    """Closest vector in three half-spaces; enumerate the eight possible active sets.

    Float64 inner products and a <=3x3 CPU solve avoid an extra QP dependency.
    Zero is a safe fallback if an ill-conditioned system defeats the active-set
    test. This solves sampled linearized constraints, not held-out IoU constraints.
    """
    rows = normalized_gradients(gradients)
    gram = (rows @ rows.T).cpu()
    dots = (rows @ proposal.double()).cpu()
    best, best_cost = None, float("inf")
    tolerance = 1e-10 * max(float(proposal.norm()), 1e-8)
    for count in range(len(rows) + 1):
        for active in combinations(range(len(rows)), count):
            multiplier = torch.zeros(len(rows), dtype=torch.float64)
            if active:
                indices = torch.tensor(active)
                matrix = gram[indices][:, indices]
                values = -(torch.linalg.pinv(matrix, rtol=1e-12) @ dots[indices])
                if bool((values < -tolerance).any()):
                    continue
                multiplier[indices] = values.clamp_min(0)
            if bool((dots + gram @ multiplier < -tolerance).any()):
                continue
            cost = float(multiplier @ gram @ multiplier)
            if cost < best_cost:
                best, best_cost = multiplier, cost
    if best is None:
        return torch.zeros_like(proposal), True
    result = proposal.double() + rows.T @ best.to(proposal.device)
    return result.to(proposal.dtype), False


@torch.no_grad()
def constrained_step(
    controls,
    scaler,
    gradients,
    *,
    project=True,
    tolerance=1e-6,
    group_weights=(0.5, 0.5, 0.0),
    additive_gradient=None,
):
    """Take an optimizer proposal, then apply a feasible realized parameter update.

    Adam moments retain proposal gradients; each future realized step is projected
    again. Overflow skips the optimizer. A rounding violation rejects weight motion
    but retains the successful optimizer proposal's moments and schedule progress.
    """
    weights = validate_group_weights(group_weights)
    if gradients.ndim != 2 or gradients.shape[0] != 3:
        raise ValueError("Expected one gradient vector for each of three groups")
    parameters = controls.parameters
    finite = torch.isfinite(gradients).all()
    if additive_gradient is not None:
        if additive_gradient.shape != gradients.shape[1:]:
            raise ValueError(
                "Additive gradient must match the flattened trainable vector"
            )
        finite = finite & torch.isfinite(additive_gradient).all()
    if dist.is_initialized():
        flag = (~finite).to(torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        finite = flag == 0
    if not bool(finite) and not scaler.is_enabled():
        raise FloatingPointError("Nonfinite group gradient")
    proposal_gradient = (
        gradients[0] * weights[0]
        + gradients[1] * weights[1]
        + gradients[2] * weights[2]
    )
    if additive_gradient is not None:
        proposal_gradient = proposal_gradient + additive_gradient
    position = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = (
            proposal_gradient[position : position + count].reshape_as(parameter)
            * scaler.get_scale()
        ).clone()
        position += count
    if not bool(finite):
        parameters[0].grad.fill_(float("nan"))
    scaler.unscale_(controls.optimizer)
    metrics = {
        "optimizer_update": 0.0,
        "overflow": 0.0,
        "projection_applied": 0.0,
        "projection_rejected": 0.0,
        "qp_zero_fallback": 0.0,
        "proposal_norm": 0.0,
        "actual_update_norm": 0.0,
        "update_ratio": 0.0,
        "min_group_cosine": 0.0,
        "conflict_fraction": 0.0,
        "gradient_cosine_small_medium": 0.0,
        "gradient_cosine_small_large": 0.0,
        "gradient_cosine_medium_large": 0.0,
    }
    if bool(finite):
        # Float64 norm avoids overflow when every individual gradient is finite.
        norm = proposal_gradient.double().norm()
        factor = min(1.0, 1.0 / (float(norm) + 1e-6))
        for parameter in parameters:
            parameter.grad.mul_(factor)
    before = parameter_vector(parameters) if bool(finite) else None
    completed = controls.completed_steps
    controls.step(scaler)
    if controls.completed_steps == completed:
        metrics["overflow"] = 1.0
        return metrics
    metrics["optimizer_update"] = 1.0
    displacement = before - parameter_vector(parameters)
    proposed_norm = float(displacement.norm())
    rows = normalized_gradients(gradients)
    gram = rows @ rows.T
    metrics.update(
        gradient_cosine_small_medium=float(gram[0, 1]),
        gradient_cosine_small_large=float(gram[0, 2]),
        gradient_cosine_medium_large=float(gram[1, 2]),
    )
    original_dots = rows @ displacement.double()
    metrics["conflict_fraction"] = float((original_dots < 0).double().mean())
    if project:
        corrected, fallback = project_displacement(displacement, gradients)
        metrics["qp_zero_fallback"] = float(fallback)
        metrics["projection_applied"] = float(not torch.equal(corrected, displacement))
        copy_parameters(parameters, before - corrected)
        # Evaluate the ACTUAL rounded FP32 movement, not just the ideal QP vector.
        displacement = before - parameter_vector(parameters)
        norm = float(displacement.norm())
        if bool(((rows @ displacement.double()) < -tolerance * max(norm, 1e-12)).any()):
            copy_parameters(parameters, before)
            displacement = torch.zeros_like(displacement)
            metrics["projection_rejected"] = 1.0
    norm = float(displacement.norm())
    metrics.update(
        proposal_norm=proposed_norm,
        actual_update_norm=norm,
        update_ratio=norm / max(proposed_norm, 1e-20),
        min_group_cosine=float((rows @ displacement.double()).min()) / max(norm, 1e-20),
    )
    return metrics
