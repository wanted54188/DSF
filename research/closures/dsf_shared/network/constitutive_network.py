"""Four prior modes, one original constitutive forward, no extra parameters.

The checked AST edit changes exactly two feature expressions and inserts one
prior selector. All remaining FULL arithmetic is compiled from the original
forward. C4 itself is imported from the previously verified implementation.
"""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import textwrap
from types import MethodType
import weakref

import torch
import spnsgs.solver as native

ROOT = Path(__file__).resolve().parents[2]
C4_PATH = ROOT/'research/ssc_clark4_dissipation_v1_20261102/clark4.py'
spec = importlib.util.spec_from_file_location('_full_integration_clark4', C4_PATH)
c4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c4)
MODES = {'FULL-C2': ('c2', 'c2'), 'FULL-Prior4': ('c2', 'c4'),
         'FULL-Input4': ('c4', 'c2'), 'FULL-C4': ('c4', 'c4')}
ORIGINAL_FORWARD = native.SPNSGSClosure.forward
SOURCE = textwrap.dedent(inspect.getsource(ORIGINAL_FORWARD))


def _compile():
    tree = ast.parse(SOURCE)
    fn = tree.body[0]
    assignments = [i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'gradient_shape_y' for t in n.targets)]
    if len(assignments) != 1:
        raise RuntimeError('Original FULL forward changed: gradient assignment')
    edits = 0
    class Features(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal edits
            if (isinstance(node.func, ast.Attribute) and node.func.attr == '_normalise'
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in ('gradient_shape_x', 'gradient_shape_y')):
                node.args[0].id = node.args[0].id.replace('gradient_shape', '_prior_feature')
                edits += 1
            return self.generic_visit(node)
    tree = Features().visit(tree)
    if edits != 2:
        raise RuntimeError('Original FULL forward changed: expected exactly two C2 feature channels')
    insertion = ast.parse('''
_prior_feature_x, _prior_feature_y, gradient_shape_x, gradient_shape_y = self._prior_vectors(
    gradient_shape_x, gradient_shape_y, omega_x, omega_y, u_x, u_y, v_x, v_y)
''').body
    fn.body[assignments[0]+1:assignments[0]+1] = insertion
    fn.name = '_integrated_forward'
    ast.fix_missing_locations(tree)
    space = dict(ORIGINAL_FORWARD.__globals__)
    exec(compile(tree, str(Path(__file__).resolve()) + ':checked_forward', 'exec'), space)
    return space[fn.name], ast.unparse(tree)


FORWARD, TRANSFORMED_SOURCE = _compile()


def _vectors(self, x2, y2, wx, wy, ux, uy, vx, vy):
    feature, constitutive = MODES[self._integration_mode]
    p2 = (x2, y2)
    if 'c4' in (feature, constitutive):
        solver = self._prior_solver_ref()
        if solver is None:
            raise RuntimeError('Prior derivative provider expired')
        state = dict(omega_x=wx, omega_y=wy, u_x=ux, u_y=uy, v_x=vx, v_y=vy)
        # Add the reused fourth-order increment to the ORIGINAL C2 arithmetic.
        inc = c4.fourth_increment(solver, state, self.filter_width)
        p4 = (x2+inc[0], y2+inc[1])
    else:
        p4 = p2
    pf, pc = (p2 if feature == 'c2' else p4), (p2 if constitutive == 'c2' else p4)
    if self._integration_capture:
        self._integration_last = dict(c2=p2, c4=p4, feature=pf, constitutive=pc)
    return (*pf, *pc)


def install(solver, mode, capture=False):
    if mode not in MODES:
        raise ValueError('Unknown prior mode: '+str(mode))
    closure = solver.closure
    if closure.feature_set != 'mixed_gradient_v3':
        raise ValueError('This study requires the audited 12-channel mixed_gradient_v3')
    if getattr(closure, '_integration_mode', None) is not None:
        raise RuntimeError('Install exactly once on a fresh solver')
    closure._integration_mode = mode
    closure._integration_capture = capture
    # A weak reference avoids registering the parent solver as a child module.
    object.__setattr__(closure, '_prior_solver_ref', weakref.ref(solver))
    closure._prior_vectors = MethodType(_vectors, closure)
    closure.forward = MethodType(FORWARD, closure)
    return solver


def transformation_identity():
    return dict(original_forward_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
                transformed_forward_sha256=hashlib.sha256(TRANSFORMED_SOURCE.encode()).hexdigest(),
                feature_replacements=2, prior_selector_insertions=1,
                added_parameters=0, C4_implementation=str(C4_PATH))


def violations(solver, fields):
    """Actual pointwise envelope, including current gate and both normalizations."""
    cl = solver.closure
    g = torch.sqrt(fields['omega_x']**2+fields['omega_y']**2+1e-16)
    support = fields['backscatter_support_norm']
    beta = cl.backscatter_factor * cl.maximum_backscatter_ratio
    if cl.experiment_variant in ('iso_dissipative', 'anisotropic_dissipative'):
        beta = 0.
    allowance = cl.maximum_local_backscatter_transfer_ratio * (
        torch.relu(fields['pi_d'])+fields['gradient_support_norm']*g)
    qbn = torch.sqrt(fields['qb_x']**2+fields['qb_y']**2)
    eigmin = fields['A_eigenvalue_min']
    return dict(signed_norm_violation=float(torch.relu(qbn-beta*support).max().detach()),
                negative_transfer_violation=float(torch.relu(-fields['pi_b']-allowance).max().detach()),
                PSD_negative_eigenvalue=float(torch.relu(-eigmin).max().detach()),
                minimum_eigenvalue=float(eigmin.min().detach()),
                trace_error=float(abs(fields['A11']+fields['A22']-2).max().detach()),
                maximum_allowance=float(allowance.max().detach()))
