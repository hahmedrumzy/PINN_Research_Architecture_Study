"""
experiment_architecture_combined.py
=====================================
Standalone, all-in-one script covering the three network-architecture
ablations: feature-dimension (trunk width), activation function, and
symmetric pooling. backbone (shared/separate) stays fixed at "separate"
and elasticity stays fixed at "linear" (paper defaults) throughout, since
those are separate axes not requested here.

Two sweep modes:

  --mode onaxis   (default) : one-at-a-time ablation. Starts from the
      baseline (hidden_dims=baseline_1024, activation=relu, pool=max)
      and varies exactly one axis per run, holding the other two at
      baseline. 6 runs total: baseline, expanded_2048, expanded_4096,
      tanh, sin, attention. Cheapest, gives you the isolated effect of
      each axis against the same baseline -- the way you'd normally
      read an ablation table.

  --mode full_grid : Cartesian product of all hidden_dims x activation x
      pool_type combinations (3 x 3 x 2 = 18 runs). Only use this if you
      specifically want to check for interaction effects between axes
      (e.g. does 'sin' behave differently at 4096-wide trunks than at
      1024-wide). Expensive -- 18 full training runs.

Hidden-dim configs:
    baseline_1024 : [1024, 512, 256, 128, 64]
    expanded_2048 : [2048, 1024, 512, 256, 128]
    expanded_4096 : [4096, 2048, 1024, 512, 256]
Activations: relu (baseline), tanh, sin
Pooling: max (baseline), attention

NOTE on 'sin': can diverge at the default Adam lr. This script auto-scales
the learning rate down for any run using 'sin' via --sin-lr-scale
(default 0.1x the base --lr) unless you pass --no-sin-lr-scale.

Usage:
    # one-at-a-time ablation (6 runs)
    python experiment_architecture_combined.py \
        --traindata-path ./toy_datasets --testdata-path ./toy_datasets \
        --outdir ./runs/architecture --device cuda --epochs 20000 --batch-size 2

    # full grid (18 runs)
    python experiment_architecture_combined.py ... --mode full_grid

    # just check one specific combo
    python experiment_architecture_combined.py ... --mode full_grid \
        --hidden-dims-subset expanded_4096 --activations-subset sin \
        --poolings-subset attention
"""

# ===================================================================== #
#  Core network + losses + dataset, v2: configurable activation +
#  configurable symmetric pooling function
# ===================================================================== #

import os
import time
import numpy as np
import torch
import torch.utils.data


def flatten(x):
    return x.view(x.size(0), -1)


class Sin(torch.nn.Module):
    """ sin(x) activation -- not built into torch.nn, needed for SIREN-style PINNs. """
    def forward(self, x):
        return torch.sin(x)


ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sin": Sin,
}


def mlp_layers(nch_input, nch_layers, b_shared=True, dropout=0.0, activation_cls=torch.nn.ReLU):
    layers = []
    last = nch_input
    for outp in nch_layers:
        weights = torch.nn.Conv1d(last, outp, 1) if b_shared else torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(activation_cls())
        if not b_shared and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


def mlp_layers_wo_relu(nch_input, nch_layers, b_shared=True, dropout=0.0):
    # final regression heads: no activation regardless of trunk activation choice
    # (matches original paper code -- output layer is linear)
    layers = []
    last = nch_input
    for outp in nch_layers:
        weights = torch.nn.Conv1d(last, outp, 1) if b_shared else torch.nn.Linear(last, outp)
        layers.append(weights)
        if not b_shared and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class MLPNet(torch.nn.Module):
    def __init__(self, nch_input, nch_layers, b_shared=True, dropout=0.0, activation_cls=torch.nn.ReLU):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *mlp_layers(nch_input, nch_layers, b_shared, dropout, activation_cls))

    def forward(self, inp):
        return self.layers(inp)


class MLPNet_wo_relu(torch.nn.Module):
    def __init__(self, nch_input, nch_layers, b_shared=True, dropout=0.0):
        super().__init__()
        self.layers = torch.nn.Sequential(*mlp_layers_wo_relu(nch_input, nch_layers, b_shared, dropout))

    def forward(self, inp):
        return self.layers(inp)


def symfn_max(x):
    """ [B, K, N] -> [B, K, 1] """
    return torch.nn.functional.max_pool1d(x, x.size(-1))


class AttentionPool(torch.nn.Module):
    """
    Learned attention-weighted pooling, alternative to hard max-pool.
    score_i = w^T x_i  (per-point scalar score via 1x1 conv)
    alpha = softmax_i(score_i)
    output = sum_i alpha_i * x_i   -> [B, K, 1]
    """
    def __init__(self, dim_k):
        super().__init__()
        self.score = torch.nn.Conv1d(dim_k, 1, 1)

    def forward(self, x):
        # x: [B, K, N]
        scores = self.score(x)                      # [B, 1, N]
        alpha = torch.softmax(scores, dim=-1)        # [B, 1, N]
        pooled = torch.sum(x * alpha, dim=-1, keepdim=True)  # [B, K, 1]
        return pooled


class TNet(torch.nn.Module):
    def __init__(self, K, activation_cls=torch.nn.ReLU):
        super().__init__()
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True, activation_cls=activation_cls))
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False, activation_cls=activation_cls))
        self.lin = torch.nn.Linear(256, K * K)
        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp):
        K, N = inp.size(1), inp.size(2)
        eye = torch.eye(K).unsqueeze(0).to(inp)
        x = self.mlp1(inp)
        x = flatten(torch.nn.functional.max_pool1d(x, N))  # TNet pooling always max -- it's a fixed
        x = self.mlp2(x)                                    # alignment sub-module, not the ablation target
        x = self.lin(x)
        return x.view(-1, K, K) + eye


class PointNetFeatures(torch.nn.Module):
    def __init__(self, dim_k=1024, use_tnet=True, scale=1, activation_cls=torch.nn.ReLU, pool_type="max"):
        super().__init__()
        mlp_h1 = [int(64 / scale), int(64 / scale)]
        mlp_h2 = [int(64 / scale), int(128 / scale), int(dim_k / scale)]
        self.h1 = MLPNet(4, mlp_h1, b_shared=True, activation_cls=activation_cls).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True, activation_cls=activation_cls).layers
        self.tnet1 = TNet(4, activation_cls=activation_cls) if use_tnet else None
        self.tnet2 = TNet(mlp_h1[-1], activation_cls=activation_cls) if use_tnet else None

        self.pool_type = pool_type
        if pool_type == "max":
            self.pool = symfn_max
        elif pool_type == "attention":
            self.pool = AttentionPool(int(dim_k / scale))
        else:
            raise ValueError(f"unknown pool_type {pool_type}")

    def forward(self, points):
        x = points.transpose(1, 2)
        if self.tnet1:
            x = self.tnet1(x).bmm(x)
        x = self.h1(x)
        if self.tnet2:
            x = self.tnet2(x).bmm(x)
        x = self.h2(x)
        return flatten(self.pool(x))


def chamfer_loss(x, y, ps):
    A, B = x, y
    r = torch.sum(A * A, dim=2).unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2).unsqueeze(-1)
    t = (r.repeat(1, 1, ps) - 2 * torch.bmm(A, B.permute(0, 2, 1))
         + r1.permute(0, 2, 1).repeat(1, ps, 1))
    d1, _ = t.min(dim=1)
    d2, _ = t.min(dim=2)
    return ((d1 + d2) / 2).mean()


def check_derivative(der, t):
    if der is None:
        return torch.zeros_like(t, requires_grad=True)
    der.requires_grad_()
    return der


def _homogeneous(points):
    ones = torch.ones(points.size(0), points.size(1), 1, dtype=points.dtype, device=points.device)
    return torch.cat((points, ones), -1)


def get_lame_vectors(source_point_set, lame1_soft, lame2_soft, lame1_stiff, lame2_stiff,
                      split_axis=2, split_frac=2 / 3):
    device = source_point_set.device
    lame1_soft = torch.as_tensor(lame1_soft, device=device, dtype=source_point_set.dtype)
    lame2_soft = torch.as_tensor(lame2_soft, device=device, dtype=source_point_set.dtype)
    lame1_stiff = torch.as_tensor(lame1_stiff, device=device, dtype=source_point_set.dtype)
    lame2_stiff = torch.as_tensor(lame2_stiff, device=device, dtype=source_point_set.dtype)

    coord = source_point_set[:, split_axis]
    coord_sorted, _ = torch.sort(coord)
    span = coord_sorted[-1] - coord_sorted[0]
    threshold = coord_sorted[0] + split_frac * span
    is_near = (coord <= threshold).unsqueeze(-1)

    lame1_vector = torch.where(is_near, lame1_stiff, lame1_soft)
    lame2_vector = torch.where(is_near, lame2_stiff, lame2_soft)
    return lame1_vector, lame2_vector


def linear_elasticity_loss(Sxx, Syy, Szz, Sxy, Sxz, Syz,
                            Exx, Eyy, Ezz, Exy, Exz, Eyz,
                            mb1, mb2, mb3, lame1_vector, lame2_vector):
    device = Sxx.device
    num_points = Sxx.shape[0]
    zero_vector = torch.zeros((num_points, 1), device=device)

    C11 = (2 * lame2_vector + lame1_vector).clone().detach().to(device)
    C12 = lame1_vector.clone().detach().to(device)
    C33 = (2 * lame2_vector).clone().detach().to(device)

    mse = torch.nn.MSELoss()
    momentum_balance = mse(mb1, zero_vector) + mse(mb2, zero_vector) + mse(mb3, zero_vector)

    ge1 = mse(Sxx - (C11 * Exx + C12 * Eyy + C12 * Ezz), zero_vector)
    ge2 = mse(Syy - (C12 * Exx + C11 * Eyy + C12 * Ezz), zero_vector)
    ge3 = mse(Szz - (C12 * Exx + C12 * Eyy + C11 * Ezz), zero_vector)
    ge4 = mse(Sxy - C33 * Exy, zero_vector)
    ge5 = mse(Sxz - C33 * Exz, zero_vector)
    ge6 = mse(Syz - C33 * Eyz, zero_vector)

    elastic_energy = 0.5 * torch.sum(
        Exx * Sxx + Eyy * Syy + Ezz * Szz + 2.0 * Exy * Sxy + 2.0 * Exz * Sxz + 2.0 * Eyz * Syz
    ) / num_points

    overall = ge1 + ge2 + ge3 + ge4 + ge5 + ge6 + elastic_energy + momentum_balance
    return overall, ge1, ge2, ge3, ge4, ge5, ge6, elastic_energy, momentum_balance


class PhysicsInformedRegistration(torch.nn.Module):
    def __init__(self, dim_k=1024, hidden_dims=(1024, 512, 256, 128, 64),
                 shared_backbone=False, activation="relu", pool_type="max"):
        super().__init__()
        self.dim_k = dim_k
        self.shared_backbone = shared_backbone
        activation_cls = ACTIVATIONS[activation]
        hidden_dims = list(hidden_dims)

        self.ptfeatures = PointNetFeatures(dim_k=dim_k, use_tnet=True,
                                            activation_cls=activation_cls, pool_type=pool_type)
        if not shared_backbone:
            self.ptfeatures_pinns = PointNetFeatures(dim_k=dim_k, use_tnet=True,
                                                       activation_cls=activation_cls, pool_type=pool_type)

        trunk_in = 2 * dim_k + 3
        self.list_layers2 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.last_layer = MLPNet_wo_relu(hidden_dims[-1], [3], b_shared=True).layers

        stress_out = hidden_dims[-1]
        self.pinns_sxx1 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.pinns_sxx2 = MLPNet_wo_relu(stress_out, [1], b_shared=True).layers
        self.pinns_syy1 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.pinns_syy2 = MLPNet_wo_relu(stress_out, [1], b_shared=True).layers
        self.pinns_szz1 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.pinns_szz2 = MLPNet_wo_relu(stress_out, [1], b_shared=True).layers
        self.pinns_sxy1 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.pinns_sxy2 = MLPNet_wo_relu(stress_out, [1], b_shared=True).layers
        self.pinns_sxz1 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.pinns_sxz2 = MLPNet_wo_relu(stress_out, [1], b_shared=True).layers
        self.pinns_syz1 = MLPNet(trunk_in, hidden_dims, b_shared=True, activation_cls=activation_cls).layers
        self.pinns_syz2 = MLPNet_wo_relu(stress_out, [1], b_shared=True).layers

    def _stress_backbone(self):
        return self.ptfeatures if self.shared_backbone else self.ptfeatures_pinns

    def forward(self, data, return_grad_tensor=False):
        device = data.device
        source = data[:, 0, :, :]
        target = data[:, 1, :, :]

        x = source[:, :, 0].reshape(-1, 1)
        y = source[:, :, 1].reshape(-1, 1)
        z = source[:, :, 2].reshape(-1, 1)
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        batch_number, num_points = data.shape[0], source.shape[1]
        source_used = torch.zeros(batch_number, num_points, 3, device=device)
        source_used[:, :, 0] = x.reshape(batch_number, -1)
        source_used[:, :, 1] = y.reshape(batch_number, -1)
        source_used[:, :, 2] = z.reshape(batch_number, -1)

        source_four = _homogeneous(source_used)
        target_four = _homogeneous(target)

        pf_src = self.ptfeatures(source_four)
        pf_tgt = self.ptfeatures(target_four)
        g = torch.cat((pf_src, pf_tgt), -1)
        g_rep = g.unsqueeze(1).repeat(1, num_points, 1)
        g_cat = torch.cat((g_rep, source), -1)
        disp_pre = self.list_layers2(g_cat.permute(0, 2, 1))
        disp = self.last_layer(disp_pre).permute(0, 2, 1)
        deformed_source = source + disp

        Ux = disp[:, :, 0].reshape(-1, 1)
        Uy = disp[:, :, 1].reshape(-1, 1)
        Uz = disp[:, :, 2].reshape(-1, 1)

        stress_backbone = self._stress_backbone()
        pf_src_s = stress_backbone(source_four)
        pf_tgt_s = stress_backbone(target_four)
        g_s = torch.cat((pf_src_s, pf_tgt_s), -1)
        g_s_rep = g_s.unsqueeze(1).repeat(1, num_points, 1)
        g_s_cat = torch.cat((g_s_rep, source), -1).permute(0, 2, 1)

        Sxx = self.pinns_sxx2(self.pinns_sxx1(g_s_cat)).reshape(-1, 1)
        Syy = self.pinns_syy2(self.pinns_syy1(g_s_cat)).reshape(-1, 1)
        Szz = self.pinns_szz2(self.pinns_szz1(g_s_cat)).reshape(-1, 1)
        Sxy = self.pinns_sxy2(self.pinns_sxy1(g_s_cat)).reshape(-1, 1)
        Sxz = self.pinns_sxz2(self.pinns_sxz1(g_s_cat)).reshape(-1, 1)
        Syz = self.pinns_syz2(self.pinns_syz1(g_s_cat)).reshape(-1, 1)

        dUx_dx = check_derivative(torch.autograd.grad(Ux, x, torch.ones_like(Ux), create_graph=True, allow_unused=True)[0], x)
        dUy_dy = check_derivative(torch.autograd.grad(Uy, y, torch.ones_like(Uy), create_graph=True, allow_unused=True)[0], y)
        dUz_dz = check_derivative(torch.autograd.grad(Uz, z, torch.ones_like(Uz), create_graph=True, allow_unused=True)[0], z)
        dUx_dy = check_derivative(torch.autograd.grad(Ux, y, torch.ones_like(Ux), create_graph=True, allow_unused=True)[0], y)
        dUy_dx = check_derivative(torch.autograd.grad(Uy, x, torch.ones_like(Uy), create_graph=True, allow_unused=True)[0], x)
        dUx_dz = check_derivative(torch.autograd.grad(Ux, z, torch.ones_like(Ux), create_graph=True, allow_unused=True)[0], z)
        dUz_dx = check_derivative(torch.autograd.grad(Uz, x, torch.ones_like(Uz), create_graph=True, allow_unused=True)[0], x)
        dUy_dz = check_derivative(torch.autograd.grad(Uy, z, torch.ones_like(Uy), create_graph=True, allow_unused=True)[0], z)
        dUz_dy = check_derivative(torch.autograd.grad(Uz, y, torch.ones_like(Uz), create_graph=True, allow_unused=True)[0], y)

        Exx, Eyy, Ezz = dUx_dx, dUy_dy, dUz_dz
        Exy = 0.5 * (dUx_dy + dUy_dx)
        Exz = 0.5 * (dUx_dz + dUz_dx)
        Eyz = 0.5 * (dUy_dz + dUz_dy)

        mb1 = (check_derivative(torch.autograd.grad(Sxx, x, torch.ones_like(Sxx), create_graph=True, allow_unused=True)[0], x)
               + check_derivative(torch.autograd.grad(Sxy, y, torch.ones_like(Sxy), create_graph=True, allow_unused=True)[0], y)
               + check_derivative(torch.autograd.grad(Sxz, z, torch.ones_like(Sxz), create_graph=True, allow_unused=True)[0], z))
        mb2 = (check_derivative(torch.autograd.grad(Sxy, x, torch.ones_like(Sxy), create_graph=True, allow_unused=True)[0], x)
               + check_derivative(torch.autograd.grad(Syy, y, torch.ones_like(Syy), create_graph=True, allow_unused=True)[0], y)
               + check_derivative(torch.autograd.grad(Syz, z, torch.ones_like(Syz), create_graph=True, allow_unused=True)[0], z))
        mb3 = (check_derivative(torch.autograd.grad(Sxz, x, torch.ones_like(Sxz), create_graph=True, allow_unused=True)[0], x)
               + check_derivative(torch.autograd.grad(Syz, y, torch.ones_like(Syz), create_graph=True, allow_unused=True)[0], y)
               + check_derivative(torch.autograd.grad(Szz, z, torch.ones_like(Szz), create_graph=True, allow_unused=True)[0], z))

        return {
            "deformed_source": deformed_source,
            "Ux": Ux, "Uy": Uy, "Uz": Uz,
            "Sxx": Sxx, "Syy": Syy, "Szz": Szz, "Sxy": Sxy, "Sxz": Sxz, "Syz": Syz,
            "Exx": Exx, "Eyy": Eyy, "Ezz": Ezz, "Exy": Exy, "Exz": Exz, "Eyz": Eyz,
            "mb1": mb1, "mb2": mb2, "mb3": mb3,
        }


def get_sample(data_dir):
    return [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]


def convert_list_into_array(sample_return):
    n = sample_return[0].shape[0]
    out = np.zeros([2, n, 3])
    out[0, :, :] = sample_return[0]
    out[1, :, :] = sample_return[1]
    return out


class prostateset_v2(torch.utils.data.Dataset):
    def __init__(self, ALL_SAMPLES, transform=None):
        self.samples = ALL_SAMPLES
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        sample_return = [sample[0], sample[1]]
        out = convert_list_into_array(sample_return)
        return np.ascontiguousarray(out, dtype=np.float32)


def load_dataset(data_dir):
    paths = get_sample(data_dir)
    all_samples = [np.load(p, allow_pickle=True) for p in paths]
    return prostateset_v2(all_samples)


def train_one_epoch(model, loader, optimizer, device, chamfer_weight, lame_args):
    model.train()
    vloss, vloss_chamfer, vloss_pinns = 0.0, 0.0, 0.0
    count = 0
    for data in loader:
        data = data.float().to(device)
        optimizer.zero_grad()
        out = model(data)

        loss_cham = chamfer_loss(out["deformed_source"], data[:, 1, :, :], data.shape[2])

        source_flat = data[:, 0, :, :].reshape(-1, 3)
        lame1_vec, lame2_vec = get_lame_vectors(
            source_flat, lame_args["lame1_soft"], lame_args["lame2_soft"],
            lame_args["lame1_stiff"], lame_args["lame2_stiff"],
        )

        pinn_out = linear_elasticity_loss(
            out["Sxx"], out["Syy"], out["Szz"], out["Sxy"], out["Sxz"], out["Syz"],
            out["Exx"], out["Eyy"], out["Ezz"], out["Exy"], out["Exz"], out["Eyz"],
            out["mb1"], out["mb2"], out["mb3"], lame1_vec, lame2_vec,
        )
        pinns_loss = pinn_out[0]
        total_loss = chamfer_weight * loss_cham + pinns_loss

        if torch.isnan(total_loss):
            raise RuntimeError(
                "Loss is NaN -- this can happen with 'sin' activation at high learning "
                "rates or poor init scale. Try a smaller --lr for the sin config."
            )

        total_loss.backward(retain_graph=True)
        optimizer.step()

        vloss += total_loss.item()
        vloss_chamfer += loss_cham.item()
        vloss_pinns += pinns_loss.item()
        count += 1

    return vloss / count, vloss_chamfer / count, vloss_pinns / count


# ===================================================================== #
#  Sweep: hidden_dims x activation x pooling
# ===================================================================== #

import itertools

HIDDEN_DIM_CONFIGS = {
    "baseline_1024": [1024, 512, 256, 128, 64],
    "expanded_2048": [2048, 1024, 512, 256, 128],
    "expanded_4096": [4096, 2048, 1024, 512, 256],
}
ACTIVATION_NAMES = ["relu", "tanh", "sin"]
POOL_TYPES = ["max", "attention"]

BASELINE = dict(hidden_dims_name="baseline_1024", activation="relu", pool_type="max")


def build_onaxis_configs(hidden_dims_subset, activations_subset, poolings_subset):
    """ One-at-a-time ablation: baseline + one variant per axis. """
    configs = []
    seen = set()

    def add(hd_name, act, pool):
        key = (hd_name, act, pool)
        if key not in seen:
            seen.add(key)
            configs.append({"name": f"hd={hd_name}_act={act}_pool={pool}",
                             "hidden_dims_name": hd_name, "activation": act, "pool_type": pool})

    add(BASELINE["hidden_dims_name"], BASELINE["activation"], BASELINE["pool_type"])
    for hd_name in hidden_dims_subset:
        if hd_name != BASELINE["hidden_dims_name"]:
            add(hd_name, BASELINE["activation"], BASELINE["pool_type"])
    for act in activations_subset:
        if act != BASELINE["activation"]:
            add(BASELINE["hidden_dims_name"], act, BASELINE["pool_type"])
    for pool in poolings_subset:
        if pool != BASELINE["pool_type"]:
            add(BASELINE["hidden_dims_name"], BASELINE["activation"], pool)
    return configs


def build_full_grid_configs(hidden_dims_subset, activations_subset, poolings_subset):
    configs = []
    for hd_name, act, pool in itertools.product(hidden_dims_subset, activations_subset, poolings_subset):
        configs.append({"name": f"hd={hd_name}_act={act}_pool={pool}",
                         "hidden_dims_name": hd_name, "activation": act, "pool_type": pool})
    return configs


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traindata-path", required=True)
    p.add_argument("--testdata-path", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dim-k", type=int, default=1024)
    p.add_argument("--backbone", choices=["shared", "separate"], default="separate")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sin-lr-scale", type=float, default=0.1,
                    help="lr multiplier applied only to runs using 'sin' activation")
    p.add_argument("--no-sin-lr-scale", action="store_true",
                    help="disable the automatic lr scale-down for 'sin' runs")
    p.add_argument("--chamfer-weight", type=float, default=10.0)
    p.add_argument("--lame1-soft", type=float, default=82.21)
    p.add_argument("--lame2-soft", type=float, default=1.68)
    p.add_argument("--lame1-stiff", type=float, default=82.21)
    p.add_argument("--lame2-stiff", type=float, default=1.68)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--mode", choices=["onaxis", "full_grid"], default="onaxis")
    p.add_argument("--hidden-dims-subset", nargs="+", default=list(HIDDEN_DIM_CONFIGS.keys()),
                    choices=list(HIDDEN_DIM_CONFIGS.keys()))
    p.add_argument("--activations-subset", nargs="+", default=ACTIVATION_NAMES, choices=ACTIVATION_NAMES)
    p.add_argument("--poolings-subset", nargs="+", default=POOL_TYPES, choices=POOL_TYPES)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    trainset = load_dataset(args.traindata_path)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    lame_args = dict(lame1_soft=args.lame1_soft, lame2_soft=args.lame2_soft,
                      lame1_stiff=args.lame1_stiff, lame2_stiff=args.lame2_stiff)

    if args.mode == "onaxis":
        configs = build_onaxis_configs(args.hidden_dims_subset, args.activations_subset, args.poolings_subset)
    else:
        configs = build_full_grid_configs(args.hidden_dims_subset, args.activations_subset, args.poolings_subset)

    print(f"Mode: {args.mode} | {len(configs)} configuration(s) to run\n")

    results = []
    for cfg in configs:
        hd_name, act, pool = cfg["hidden_dims_name"], cfg["activation"], cfg["pool_type"]
        hidden_dims = HIDDEN_DIM_CONFIGS[hd_name]
        lr = args.lr
        if act == "sin" and not args.no_sin_lr_scale:
            lr = args.lr * args.sin_lr_scale

        print(f"\n{'='*70}\nCONFIG: {cfg['name']}  (hidden_dims={hidden_dims}, lr={lr:g})\n{'='*70}")

        torch.manual_seed(0)
        model = PhysicsInformedRegistration(
            dim_k=args.dim_k, hidden_dims=hidden_dims,
            shared_backbone=(args.backbone == "shared"),
            activation=act, pool_type=pool).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        run_dir = os.path.join(args.outdir, cfg["name"])
        os.makedirs(run_dir, exist_ok=True)

        diverged = False
        for epoch in range(args.epochs):
            t0 = time.time()
            try:
                loss, loss_c, loss_p = train_one_epoch(
                    model, trainloader, optimizer, device, args.chamfer_weight, lame_args)
            except RuntimeError as e:
                print(f"  [{cfg['name']}] DIVERGED at epoch {epoch}: {e}")
                diverged = True
                break
            history.append((loss, loss_c, loss_p))
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                print(f"  [{cfg['name']}] epoch {epoch:5d} | total {loss:.5f} | "
                      f"chamfer {loss_c:.5f} | pinns {loss_p:.5f} | {time.time()-t0:.2f}s")
            if epoch % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(run_dir, "model_last.pth"))

        with open(os.path.join(run_dir, "history.csv"), "w") as f:
            f.write("total,chamfer,pinns\n")
            for row in history:
                f.write(",".join(f"{v:.6f}" for v in row) + "\n")

        if diverged or not history:
            results.append((cfg["name"], hd_name, act, pool, n_params,
                             float("nan"), float("nan"), float("nan")))
        else:
            final_total, final_cham, final_pinn = history[-1]
            results.append((cfg["name"], hd_name, act, pool, n_params,
                             final_total, final_cham, final_pinn))

    print(f"\n{'='*110}\nCOMPARISON TABLE\n{'='*110}")
    header = f"{'config':<38}{'hidden_dims':<16}{'activation':<12}{'pool':<12}{'#params(M)':<12}{'final_total':<14}{'final_cham':<14}{'final_pinn':<14}"
    print(header)
    for name, hd, act, pool, n_params, ft, fc, fp in results:
        print(f"{name:<38}{hd:<16}{act:<12}{pool:<12}{n_params/1e6:<12.2f}{ft:<14.5f}{fc:<14.5f}{fp:<14.5f}")

    with open(os.path.join(args.outdir, "comparison.csv"), "w") as f:
        f.write("config,hidden_dims,activation,pool,n_params,final_total,final_chamfer,final_pinns\n")
        for name, hd, act, pool, n_params, ft, fc, fp in results:
            f.write(f"{name},{hd},{act},{pool},{n_params},{ft:.6f},{fc:.6f},{fp:.6f}\n")
    print(f"\nSaved comparison table to {os.path.join(args.outdir, 'comparison.csv')}")


if __name__ == "__main__":
    main()