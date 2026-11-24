"""C4 group convolutions and a typed decoder for the production FULL constitutive law."""

import ast
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def spatial(field, turn):
    """Active rotation on the periodic grid, about the index origin."""
    turn %= 4
    if turn == 0:
        return field
    rotated = torch.rot90(field, -turn, dims=(-2, -1))
    shifts = (int(turn in (2, 3)), int(turn in (1, 2)))
    return torch.roll(rotated, shifts, dims=(-2, -1))


def components(x, y, turn):
    """Rotate Cartesian components already evaluated at the rotated location."""
    return ((x, y), (-y, x), (-x, -y), (y, -x))[turn % 4]


def vector(x, y, turn):
    return components(spatial(x, turn), spatial(y, turn), turn)


class C4Conv(nn.Module):
    """Lift typed inputs or mix regular representations with one tied kernel."""

    def __init__(self, in_channels, out_channels, dilation=1, lift=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.lift = lift
        groups = 1 if lift else 4
        self.weight = nn.Parameter(torch.empty(out_channels, groups * in_channels, 3, 3))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(groups * in_channels * 9)
        nn.init.uniform_(self.bias, -bound, bound)

    def expanded_weight(self):
        kernels = []
        for turn in range(4):
            # Kernel offsets are centered; their rotation needs no periodic shift.
            weight = torch.rot90(self.weight, -turn, dims=(-2, -1))
            if self.lift:
                gx, gy = components(weight[:, 2], weight[:, 3], turn)
                px, py = components(weight[:, 10], weight[:, 11], turn)
                weight = torch.cat((weight[:, :2], gx[:, None], gy[:, None], weight[:, 4:10],
                                    px[:, None], py[:, None]), dim=1)
            else:
                # At output orientation o, input i uses the kernel for i-o.
                weight = weight.reshape(self.out_channels, 4, self.in_channels, 3, 3)
                weight = weight.roll(turn, dims=1).flatten(1, 2)
            kernels.append(weight)
        return torch.cat(kernels, dim=0)

    def forward(self, x):
        padded = F.pad(x, (self.dilation,) * 4, mode="circular")
        return F.conv2d(padded, self.expanded_weight(), self.bias.repeat(4), dilation=self.dilation)


class C4Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.width = width
        self.conv1 = C4Conv(width, width, dilation)
        self.conv2 = C4Conv(width, width, dilation)
        self.norm1 = nn.GroupNorm(4, width)
        self.norm2 = nn.GroupNorm(4, width)

    def normalise(self, x, norm):
        batch, _, height, width = x.shape
        regular = x.reshape(batch, 4, self.width, height, width)
        # GN statistics cover the orbit and its affine map is shared across it.
        normalised = norm(regular.transpose(1, 2).contiguous())
        return normalised.transpose(1, 2).reshape_as(x)

    def forward(self, x):
        hidden = F.silu(self.normalise(self.conv1(x), self.norm1))
        return F.silu(x + self.normalise(self.conv2(hidden), self.norm2))


class TypedHead(nn.Module):
    """Decode c, traceless H, gate, and a signed vector at each grid point."""

    def __init__(self, width, gate_initial):
        super().__init__()
        self.conv = C4Conv(width, 6)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            self.conv.bias[3] = math.atanh(gate_initial)

    def forward(self, x):
        raw = self.conv(x).unflatten(1, (4, 6))
        oriented = []
        for turn in range(4):
            fields = raw[:, turn]
            vx, vy = components(fields[:, 4], fields[:, 5], turn)
            h = fields[:, 1:3] * ((-1) ** turn)
            oriented.append(torch.cat((fields[:, :1], h, fields[:, 3:4], vx[:, None], vy[:, None]), dim=1))
        c, h1, h2, s, vx, vy = torch.stack(oriented).mean(0).unbind(dim=1)
        # The zero slot keeps the existing signed-branch raw indices 4:7.
        return torch.stack((c, h1, h2, torch.zeros_like(c), s, vx, vy), dim=1)


def rms(*fields):
    """The RMS of a scalar, or the RMS Euclidean magnitude of a vector."""
    field = fields[0] if len(fields) == 1 else torch.stack(fields)
    return torch.linalg.vector_norm(field) / math.sqrt(fields[0].numel())


def _normalise(field):
    return field / rms(field).clamp_min(1e-6)


def _assignment_index(function, name):
    matches = [
        i for i, stmt in enumerate(function.body)
        if isinstance(stmt, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in stmt.targets)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one production assignment for {name}")
    return matches[0]


def build_forward(prior):
    """Replace only the Cholesky coordinates and restore the state RMS chain rule."""
    tree = ast.parse(prior.TRANSFORMED_SOURCE)
    fn = tree.body[0]
    start = _assignment_index(fn, "l11")
    stop = _assignment_index(fn, "anisotropy_radius")
    expected = ("l11", "l21", "l22", "a11_tilde", "a12_tilde", "a22_tilde", "trace_tilde",
                "a11_raw", "a12_raw", "a22_raw", "anisotropy_x", "anisotropy_y")
    if [stmt.targets[0].id for stmt in fn.body[start:stop]] != list(expected):
        raise RuntimeError("Production Cholesky segment changed")
    fn.body[start:stop] = ast.parse("anisotropy_x = raw[1]\nanisotropy_y = raw[2]").body
    cap = fn.body[_assignment_index(fn, "maximum_radius")]
    # Leave a few ulps for the fp32 eigenvalue reconstruction near saturation.
    cap.value = ast.BinOp(cap.value, ast.Sub(), ast.parse("8 * torch.finfo(raw.dtype).eps", mode="eval").body)
    norm = fn.body[_assignment_index(fn, "direction_norm")]
    norm.value = ast.parse("torch.linalg.vector_norm(torch.stack((direction_x, direction_y)), dim=0)", mode="eval").body

    descriptors = dict(omega_rms="rms(omega).clamp_min(1e-6)", gradient_rms="rms(omega_x, omega_y)",
                       laplacian_rms="rms(laplacian_omega)", strain_rms="rms(strain)")
    changed = set()
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name in descriptors:
                if not any(isinstance(node, ast.Attribute) and node.attr == "detach" for node in ast.walk(stmt)):
                    raise RuntimeError(f"Production RMS definition changed for {name}")
                stmt.value = ast.parse(descriptors[name], mode="eval").body
                changed.add(name)
    if changed != set(descriptors):
        raise RuntimeError("Missing production descriptors")
    namespace = dict(prior.FORWARD.__globals__, rms=rms)
    exec(compile(ast.fix_missing_locations(tree), __file__ + ":typed_FULL", "exec"), namespace)
    return namespace[fn.name]


def install(solver, prior, width):
    closure = solver.closure
    paper = solver.config.paper
    if closure._integration_mode != "FULL-C4" or closure.maximum_anisotropy_condition != 16.0:
        raise RuntimeError("Expected FULL-C4 with the original SPD cap")
    device, dtype = closure.input.weight.device, closure.input.weight.dtype
    dilations = paper.closure_dilations or [1]
    closure.input = C4Conv(12, width, lift=True).to(device=device, dtype=dtype)
    closure.blocks = nn.Sequential(*(
        C4Block(width, int(dilations[i % len(dilations)])) for i in range(paper.closure_layers)
    )).to(device=device, dtype=dtype)
    closure.output = TypedHead(width, paper.backscatter_gate_initial).to(device=device, dtype=dtype)
    closure._normalise = _normalise
    closure.forward = MethodType(build_forward(prior), closure)
    return solver
