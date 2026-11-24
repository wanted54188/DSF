"""Shared dense closure supervision, explicitly different from sparse PoF."""
import torch
from spnsgs.paper import _sign_balanced_transfer_loss


def quantities(s,cache,i,st,f,source):
    y=cache['source'][i];eps=1e-12
    mse=lambda x:torch.mean(x.square())
    source_mse=mse(source-y)/mse(y).detach().clamp_min(eps)
    x=source-source.mean();z=y-y.mean()
    corr=torch.mean(x*z)/torch.sqrt(mse(x)*mse(z)+1e-16)
    qx,qy=cache['qx'][i],cache['qy'][i]
    flux=(mse(f['qx']-qx)+mse(f['qy']-qy))/(mse(qx)+mse(qy)).detach().clamp_min(eps)
    pi=cache['pi'][i];weights=torch.full_like(pi.reshape(-1),1/pi.numel())
    signed=_sign_balanced_transfer_loss(f['pi'].reshape(-1),pi.reshape(-1),weights,
                                       s.config.paper.sparse_sgs_transfer_huber_delta)
    signed_mean=(torch.mean(f['pi']-pi)/torch.sqrt(mse(pi).detach().clamp_min(eps))).square()
    budget=source_mse*0
    for obs in (st['psi'],st['omega']):
        scale=torch.sqrt(mse(obs).detach()*mse(y).detach()).clamp_min(eps)
        budget+=((torch.mean(obs*(source-y)))/scale).square()/2
    cl=s.closure
    anisotropy=torch.relu(f['A_condition_number']/s.config.paper.anisotropy_condition_soft_limit-1).square().mean()
    sat=torch.relu(f['backscatter_ratio']/cl.maximum_backscatter_ratio-.9).square().mean()
    dsat=torch.relu(f['c_theta']/cl.dissipation_coefficient_soft_limit-.9).square().mean()
    neg=torch.relu(-f['pi'].mean())/f['pi'].abs().mean().detach().clamp_min(1e-6)
    return dict(source=source_mse,source_correlation=1-corr,flux=flux,signed_transfer=signed,
                signed_mean=signed_mean,budget=budget,anisotropy=anisotropy,signed_saturation=sat,
                dissipation_saturation=dsat,negative_mean_transfer=neg)


def total(q,weights):return sum(weights[k]*q[k] for k in weights)
