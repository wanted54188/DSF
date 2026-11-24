"""Literal branch-flux envelope test; no optimization or flux modification."""
import torch


def dissipative_certificate(g, qd, delta, cmax, condition=16., eps=1e-12):
    """Feasibility of qd=-delta^3 |g|_epsilon M g, M=cA.

    This tests the CLOSED physical envelope, not bounded Cholesky-head reachability.
    In the (e,Je) basis, M=[[a,b],[b,2c-a]]. Eigenvalue-radius <= r*c
    iff (1-r^2)c^2-2*a*c+a^2+b^2 <= 0. Existence also needs c<=cmax.
    The returned c,A witness is a diagnostic, never a repaired flux.
    """
    gm = torch.linalg.vector_norm(g, dim=0)
    gn = (g.square().sum(0)+1e-16).sqrt()
    zero = gm == 0
    e = g/torch.where(zero, torch.ones_like(gm), gm)
    e[0] = torch.where(zero, torch.ones_like(gm), e[0])
    e[1] = torch.where(zero, torch.zeros_like(gm), e[1])
    t = torch.stack((-e[1], e[0]))
    scale = delta**3*gn*gm
    scaled = -qd/torch.where(zero, torch.ones_like(scale), scale)
    a = (scaled*e).sum(0); b = (scaled*t).sum(0)
    r = (condition-1)/(condition+1)
    discriminant = r*r*a*a-(1-r*r)*b*b
    dscale = r*r*a*a+(1-r*r)*b*b
    negative_d = discriminant < -eps*dscale.clamp_min(1e-300)
    root = discriminant.clamp_min(0).sqrt()
    cminus = (a*a+b*b)/(a+root).clamp_min(1e-300)
    cplus = (a+root)/(1-r*r)
    c_bad = cminus > cmax*(1+eps)
    direction_bad = (a <= 0) | negative_d
    nonzero_at_zero = zero & (torch.linalg.vector_norm(qd, dim=0) != 0)
    bad = (~zero & (direction_bad | c_bad)) | nonzero_at_zero
    upper = torch.minimum(cplus, torch.full_like(cplus, cmax))
    c = .5*(cminus+upper)
    c = torch.where(zero, torch.full_like(c, .5*cmax), c)
    d = 2*c-a
    # Symmetric tensor witness, expressed in original x,y coordinates.
    M11 = a*e[0]**2+2*b*e[0]*t[0]+d*t[0]**2
    M12 = a*e[0]*e[1]+b*(e[0]*t[1]+t[0]*e[1])+d*t[0]*t[1]
    M22 = a*e[1]**2+2*b*e[1]*t[1]+d*t[1]**2
    M11 = torch.where(zero, c, M11); M12 = torch.where(zero, torch.zeros_like(c), M12)
    M22 = torch.where(zero, c, M22)
    A = torch.stack((M11/c.clamp_min(1e-300), M12/c.clamp_min(1e-300), M22/c.clamp_min(1e-300)))
    reconstructed = -delta**3*gn*torch.stack((M11*g[0]+M12*g[1], M12*g[0]+M22*g[1]))
    valid = ~bad
    witness_error = float(torch.linalg.vector_norm((reconstructed-qd)[:, valid])) if bool(valid.any()) else None
    pi = -(qd*g).sum(0)
    eligible = ~zero & ~direction_bad
    result = dict(qd_envelope_pass=not bool(bad.any()), qd_violation_count=int(bad.sum()),
        qd_violation_fraction=float(bad.double().mean()), qd_negative_pairing_count=int((pi < 0).sum()),
        min_pi_d=float(pi.min()), qd_zero_gradient_nonzero_count=int(nonzero_at_zero.sum()),
        qd_angle_failure_count=int((~zero & direction_bad).sum()),
        qd_coefficient_failure_count=int((~zero & ~direction_bad & c_bad).sum()),
        max_required_min_coefficient=float(cminus[eligible].max()) if bool(eligible.any()) else None,
        coefficient_maximum=cmax, witness_reconstruction_l2_on_feasible_points=witness_error)
    return result, dict(c=c, A=A, bad=bad, a=a, b=b, discriminant=discriminant)


def envelope(s, st, qd, qb, current_original, eps=1e-12):
    g = torch.stack((st['omega_x'], st['omega_y']))
    cl = s.closure
    dcheck, witness = dissipative_certificate(g, qd, cl.filter_width,
                                             cl.dissipation_coefficient_soft_limit,
                                             cl.maximum_anisotropy_condition, eps)
    gn = (g.square().sum(0)+1e-16).sqrt()
    G = current_original['gradient_support_norm']
    beta = cl.backscatter_factor*cl.maximum_backscatter_ratio
    bound = beta*(qd.square().sum(0)+G.square()+2e-16).sqrt()
    amplitude = torch.linalg.vector_norm(qb, dim=0)
    pi_d = -(qd*g).sum(0)
    injection = (qb*g).sum(0).relu()
    allowed = cl.maximum_local_backscatter_transfer_ratio*(pi_d.relu()+G*gn)
    ratio = amplitude/bound.clamp_min(1e-300)
    tratio = injection/allowed.clamp_min(1e-300)
    amp_bad = ratio > 1+eps; tr_bad = tratio > 1+eps
    result = dict(**dcheck, signed_amplitude_pass=not bool(amp_bad.any()),
        signed_transfer_pass=not bool(tr_bad.any()), max_signed_amplitude_ratio=float(ratio.max()),
        max_signed_transfer_ratio=float(tratio.max()),
        signed_amplitude_violation_count=int(amp_bad.sum()), signed_transfer_violation_count=int(tr_bad.sum()),
        max_signed_amplitude_absolute_excess=float((amplitude-bound).relu().max()),
        max_signed_transfer_absolute_excess=float((injection-allowed).relu().max()),
        max_qd_norm=float(torch.linalg.vector_norm(qd, dim=0).max()), max_qb_norm=float(amplitude.max()))
    result['passed'] = all(result[k] for k in ('qd_envelope_pass', 'signed_amplitude_pass', 'signed_transfer_pass'))
    return result, witness
