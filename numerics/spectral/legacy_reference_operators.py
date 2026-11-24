"""CPU diagnostics matching solver.py FFT conventions; no data IO or networks."""
from dataclasses import dataclass
import numpy as np


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def rel2(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-30))


@dataclass
class Fourier:
    ny: int
    nx: int
    ly: float
    lx: float
    fraction: float

    def __post_init__(self):
        if min(self.nx, self.ny) < 4 or min(self.lx, self.ly) <= 0 or not 0 < self.fraction <= 1:
            raise ValueError('Invalid periodic grid/domain/projection')
        mx=np.fft.fftfreq(self.nx, d=1/self.nx)
        my=np.fft.fftfreq(self.ny, d=1/self.ny)
        self.mx,self.my=np.meshgrid(mx,my)
        self.kx=self.mx*(2*np.pi/self.lx)
        self.ky=self.my*(2*np.pi/self.ly)
        self.k2=self.kx**2+self.ky**2
        self.mask=(np.abs(self.mx)<=self.fraction*(self.nx//2)) & (np.abs(self.my)<=self.fraction*(self.ny//2))
        self.mask[0,0]=False
        # ifft(ik*fft(real_field)).real annihilates the axis Nyquist mode.
        # Explicit effective symbols make the discrete Helmholtz inverse correct.
        self.dx=self.kx.copy();self.dy=self.ky.copy()
        if self.nx%2==0:self.dx[:,self.nx//2]=0
        if self.ny%2==0:self.dy[self.ny//2,:]=0
        self.d2=self.dx**2+self.dy**2
        self.shell=np.rint(np.hypot(self.mx,self.my)).astype(int)

    def derivative(self,a,axis):
        if axis not in ('x','y'):raise ValueError(axis)
        k=self.kx if axis=='x' else self.ky
        return np.fft.ifft2(1j*k*np.fft.fft2(a)).real

    def project(self,a):
        return np.fft.ifft2(self.mask*np.fft.fft2(a)).real

    def grad(self,a):
        return np.stack([self.derivative(a,'x'),self.derivative(a,'y')])

    def div(self,q):
        return self.derivative(q[0],'x')+self.derivative(q[1],'y')

    def source(self,q,projected=True):
        s=-self.div(q)
        return self.project(s) if projected else s

    def psi(self,omega):
        inv=np.divide(1.,self.k2,out=np.zeros_like(self.k2),where=self.k2>0)
        return np.fft.ifft2(np.fft.fft2(omega)*self.mask*inv).real

    def shell_power(self,a):
        h=np.fft.fft2(a)/a.size
        return np.bincount(self.shell.ravel(),weights=np.abs(h).ravel()**2)

    def shell_cospectrum(self,a,b):
        h=np.fft.fft2(a)/a.size;k=np.fft.fft2(b)/b.size
        return np.bincount(self.shell.ravel(),weights=np.real(h*np.conj(k)).ravel())

    def decompose(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(2,self.ny,self.nx) or not np.isfinite(q).all():raise ValueError('Expected finite q[2,ny,nx]')
        h=np.fft.fft2(q,axes=(-2,-1))
        dot=self.dx*h[0]+self.dy*h[1]
        factor=np.divide(dot,self.d2,out=np.zeros_like(dot),where=self.d2>0)
        longitudinal=np.fft.ifft2(np.stack([self.dx*factor,self.dy*factor]),axes=(-2,-1)).real
        div_free=q-longitudinal
        active=np.stack([self.project(v) for v in longitudinal])
        mean=np.broadcast_to(q.mean(axis=(-2,-1),keepdims=True),q.shape).copy()
        extra_null=np.fft.ifft2(h*(self.d2==0),axes=(-2,-1)).real-mean
        return dict(longitudinal=longitudinal,divergence_free=div_free,
                    harmonic_mean=mean,extra_discrete_null=extra_null,
                    source_active=active,solver_null=q-active,
                    rejected_longitudinal=longitudinal-active)

    def decomposition_checks(self,q):
        p=self.decompose(q);s=self.source(q)
        scale=max(rms(q),1e-30);ds=max(rms(self.div(q)),scale,1e-30)
        return dict(reconstruction_relative= rms(q-p['longitudinal']-p['divergence_free'])/scale,
                    physical_null_divergence_relative=rms(self.div(p['divergence_free']))/ds,
                    solver_null_source_relative=rms(self.source(p['solver_null']))/ds,
                    source_reconstruction_relative=rms(s-self.source(p['source_active']))/ds,
                    orthogonality_normalized=abs(float(np.mean(p['source_active']*p['solver_null'])))/scale**2,
                    active_flux_squared_fraction=float(np.sum(p['source_active']**2)/max(np.sum(q*q),1e-60)),
                    solver_null_flux_squared_fraction=float(np.sum(p['solver_null']**2)/max(np.sum(q*q),1e-60)),
                    extra_discrete_null_rms=rms(p['extra_discrete_null']))

    def budgets(self,omega,q):
        source=self.source(q);g=self.grad(self.project(omega));psi=self.psi(omega)
        pi=-np.sum(q*g,axis=0);pien=-np.sum(q*self.grad(psi),axis=0)
        z=float(np.mean(omega*source));e=float(np.mean(psi*source))
        return dict(enstrophy_source_rate=z,mean_Pi=float(pi.mean()),
                    energy_source_rate=e,mean_energy_transfer=float(pien.mean()),
                    enstrophy_identity_abs=abs(z+float(pi.mean())),
                    energy_identity_abs=abs(e+float(pien.mean())))


def paired_fields(op,omega,qf,qa,qref):
    sf,sa,sr=op.source(qf),op.source(qa),op.source(qref)
    ds=sf-sa;dq=qf-qa
    out=dict(pair_source_relative_to_reference=rms(ds)/max(rms(sr),1e-30),
             full_source_relL2_matched=rel2(sf,sr),A0B1_source_relL2_matched=rel2(sa,sr),
             full_source_relL2_physical=rel2(sf,op.source(qref,False)),
             A0B1_source_relL2_physical=rel2(sa,op.source(qref,False)),
             flux_difference_rms=rms(dq),source_difference_rms=rms(ds),
             helmholtz=op.decomposition_checks(dq),
             FULL_budget=op.budgets(omega,qf),A0B1_budget=op.budgets(omega,qa),reference_matched_budget=op.budgets(omega,qref))
    g=op.grad(op.project(omega));refpi=-np.sum(qref*g,axis=0)
    local={}
    for label,q in [('FULL',qf),('A0B1',qa)]:
        pi=-np.sum(q*g,axis=0);a=pi-pi.mean();b=refpi-refpi.mean()
        den=float(np.linalg.norm(a)*np.linalg.norm(b))
        local[label]=dict(Pi_correlation_matched=float(np.sum(a*b)/den) if den>0 else None,
                          Pi_relative_L2_matched=rel2(pi,refpi),negative_fraction=float(np.mean(pi<0)),
                          positive_transfer_integral_mean=float(np.maximum(pi,0).mean()),
                          negative_transfer_integral_magnitude_mean=float(np.maximum(-pi,0).mean()))
    out['local_transfer_matched']=local
    return out
