"""Small GPU residual-k-means and geometry-aware trajectory encoder."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch


def wrap(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


def to_diff(poses):
    previous = torch.cat([torch.zeros_like(poses[..., :1, :]), poses[..., :-1, :]], -2)
    d = poses - previous
    c, s = previous[..., 2].cos(), previous[..., 2].sin()
    return torch.stack([c * d[..., 0] + s * d[..., 1],
                        -s * d[..., 0] + c * d[..., 1], wrap(d[..., 2])], -1)


def compose(d):
    angle = d[..., 2].cumsum(-1)
    previous = angle - d[..., 2]
    c, s = previous.cos(), previous.sin()
    return torch.stack([(c * d[..., 0] - s * d[..., 1]).cumsum(-1),
                        (s * d[..., 0] + c * d[..., 1]).cumsum(-1), wrap(angle)], -1)


def features(poses, config):
    rep = config.get('representation', 'diff')
    data = to_diff(poses) if rep == 'diff' else poses
    if rep == 'sincos':
        data = torch.stack([data[..., 0], data[..., 1], data[..., 2].sin(), data[..., 2].cos()], -1)
    scale = torch.as_tensor(config['scale'], device=poses.device, dtype=poses.dtype)
    return data.flatten(-2) * scale


def decode(vector, config):
    rep, h = config.get('representation', 'diff'), config['horizon']
    scale = torch.as_tensor(config['scale'], device=vector.device, dtype=vector.dtype)
    p = (vector / scale).reshape(*vector.shape[:-1], h, 4 if rep == 'sincos' else 3)
    if rep == 'diff':
        return compose(p)
    if rep == 'sincos':
        return torch.stack([p[..., 0], p[..., 1], torch.atan2(p[..., 2], p[..., 3])], -1)
    return torch.stack([p[..., 0], p[..., 1], wrap(p[..., 2])], -1)


def distances(x, centers):
    return (x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * x @ centers.T).clamp_min_(0)


@torch.inference_mode()
def kmeans(x, k=256, iterations=35, seed=0):
    rng = torch.Generator(device=x.device).manual_seed(seed)
    centers = [x[torch.randint(len(x), (1,), device=x.device, generator=rng)].squeeze(0)]
    nearest = (x - centers[0]).square().sum(-1)
    for _ in range(1, k):
        # k-means++ avoids spending most centers on the dominant straight motions.
        idx = torch.multinomial(nearest.clamp_min(1e-20), 1, generator=rng)
        point = x[idx].squeeze(0)
        centers.append(point)
        nearest = torch.minimum(nearest, (x - point).square().sum(-1))
    centers = torch.stack(centers)
    last = float('inf')
    for iteration in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, device=x.device)
        errors, labels = [], []
        for batch in x.split(32768):
            d, ids = distances(batch, centers).min(-1)
            sums.index_add_(0, ids, batch)
            counts.index_add_(0, ids, torch.ones_like(ids, dtype=x.dtype))
            errors.append(d); labels.append(ids)
        errors, labels = torch.cat(errors), torch.cat(labels)
        missing = counts == 0
        centers = sums / counts.clamp_min(1)[:, None]
        if missing.any():
            centers[missing] = x[errors.topk(int(missing.sum())).indices]
        loss = errors.mean().item()
        if iteration >= 8 and abs(last - loss) <= max(last, 1e-10) * 1e-5:
            break
        last = loss
    if not torch.isfinite(centers).all():
        raise ValueError('nonfinite k-means centers')
    return centers, {'inertia_per_window': loss, 'iterations': iteration + 1,
                     'occupied': int((counts > 0).sum()), 'seed': seed}


@torch.inference_mode()
def fit(target, config, output, levels=3, restarts=2, iterations=35, seed=20261015):
    residual = features(target, config)
    books, logs = [], []
    for level in range(levels):
        best = None
        for restart in range(restarts):
            cb, info = kmeans(residual, iterations=iterations, seed=seed + level * 1009 + restart)
            if best is None or info['inertia_per_window'] < best[1]['inertia_per_window']:
                best = cb, info
        cb, info = best
        ids = torch.cat([distances(batch, cb).argmin(-1) for batch in residual.split(32768)])
        residual = residual - cb[ids]
        books.append(cb.cpu().numpy()); logs.append(info)
        print(f'{Path(output).stem} L{level + 1} residual_mse={residual.square().mean().item():.8g}', flush=True)
    np.savez(output, codebooks=np.stack(books), config=np.array(json.dumps(config)), training=np.array(json.dumps(logs)))
    return logs


def load(path, device='cuda', levels=None):
    with np.load(path) as data:
        cfg = json.loads(str(data['config']))
        cb = torch.as_tensor(data['codebooks'][:levels], device=device)
    return cb, cfg


def official(directory, horizon=10, device='cuda'):
    directory = Path(directory)
    weights = np.load(directory / 'jacobian_weights.npy')
    books = np.stack([np.load(directory / f'codebook_l{level}.npy') for level in range(3)])
    # Scaling each codeword before summation preserves the fixed official decoder.
    books = (books / weights)[:, :, :horizon * 3]
    cfg = {'horizon': horizon, 'representation': 'diff', 'scale': [1.] * (horizon * 3),
           'stop_l0': 6, 'fixed_official': True}
    return torch.as_tensor(books, device=device), cfg


def objective(decoded, target, valid, action_index, spec):
    # decoded is [N, candidates, H, 3]; target and masks have no candidate axis.
    xy = torch.linalg.vector_norm(decoded[..., :2] - target[:, None, :, :2], dim=-1)
    yaw = wrap(decoded[..., 2] - target[:, None, :, 2]).abs()
    weights = valid.to(xy.dtype)
    ade = (xy * weights[:, None]).sum(-1) / weights.sum(-1).clamp_min(1)[:, None]
    mean_yaw = (yaw * weights[:, None]).sum(-1) / weights.sum(-1).clamp_min(1)[:, None]
    last = (valid.sum(-1) - 1).long()
    final = yaw.gather(-1, last[:, None, None].expand(-1, yaw.shape[1], 1)).squeeze(-1)
    action_yaw = yaw.gather(-1, action_index[:, None, None].expand(-1, yaw.shape[1], 1)).squeeze(-1)
    action_xy = xy.gather(-1, action_index[:, None, None].expand(-1, xy.shape[1], 1)).squeeze(-1)
    return (spec.get('ade', 1.) * ade + spec.get('final_yaw', 0.) * final +
            spec.get('mean_yaw', 0.) * mean_yaw + spec.get('action_yaw', 0.) * action_yaw +
            spec.get('action_xy', 0.) * action_xy)


@torch.inference_mode()
def encode(target, valid, action_index, books, config, spec, beam_width=16, batch_size=128,
           fallback_codes=None, refine=0):
    results, codes_out = [], []
    levels = len(books)
    for begin in range(0, len(target), batch_size):
        stop = min(begin + batch_size, len(target))
        t, v, ai = target[begin:stop], valid[begin:stop], action_index[begin:stop]
        n = len(t)
        recon = torch.zeros(n, 1, books.shape[-1], device=t.device)
        paths = torch.empty(n, 1, 0, dtype=torch.long, device=t.device)
        greedy_recon, greedy_path = torch.zeros_like(recon), []
        # Retain the complete greedy solution, matching the released encoder's safeguard.
        for level, book in enumerate(books):
            greedy_candidates = greedy_recon + book[None]
            gs = objective(decode(greedy_candidates, config), t, v, ai, spec)
            if level == 0 and 'stop_l0' in config:
                gs[:, config['stop_l0']] = torch.inf
            gi = gs.argmin(-1)
            greedy_path.append(gi)
            greedy_recon += book[gi, None]
            candidates = (recon[:, :, None] + book[None, None]).flatten(1, 2)
            score = objective(decode(candidates, config), t, v, ai, spec)
            if level == 0 and 'stop_l0' in config:
                score[:, config['stop_l0']] = torch.inf
            chosen = score.topk(min(beam_width, score.shape[1]), largest=False, sorted=True).indices
            parents = chosen // len(book)
            paths = torch.cat([paths.gather(1, parents[..., None].expand(-1, -1, paths.shape[-1])),
                               (chosen % len(book))[..., None]], -1)
            recon = candidates.gather(1, chosen[..., None].expand(-1, -1, candidates.shape[-1]))
        greedy_path = torch.stack(greedy_path, -1)
        pool = torch.cat([recon[:, :1], greedy_recon], 1)
        pool_codes = torch.cat([paths[:, :1], greedy_path[:, None]], 1)
        if fallback_codes is not None:
            fc = fallback_codes[begin:stop]
            fv = sum(book[fc[:, i]] for i, book in enumerate(books))
            pool = torch.cat([pool, fv[:, None]], 1)
            pool_codes = torch.cat([pool_codes, fc[:, None]], 1)
        choice = objective(decode(pool, config), t, v, ai, spec).argmin(-1)
        code = pool_codes[torch.arange(n, device=t.device), choice]
        vector = pool[torch.arange(n, device=t.device), choice]
        for _ in range(refine):
            for level, book in enumerate(books):
                residual = vector - book[code[:, level]]
                candidates = residual[:, None] + book[None]
                score = objective(decode(candidates, config), t, v, ai, spec)
                if level == 0 and 'stop_l0' in config:
                    score[:, config['stop_l0']] = torch.inf
                selected = score.argmin(-1)
                code[:, level] = selected
                vector = residual + book[selected]
        results.append(decode(vector, config).cpu().numpy())
        codes_out.append(code.cpu().numpy().astype(np.int16))
    return np.concatenate(results), np.concatenate(codes_out)


@torch.inference_mode()
def encode_l2(target, books, config, batch_size=16384):
    results, codes_out = [], []
    for t in target.split(batch_size):
        residual = features(t, config)
        recon = torch.zeros_like(residual)
        paths = []
        for level, book in enumerate(books):
            score = distances(residual, book)
            if level == 0 and 'stop_l0' in config:
                score[:, config['stop_l0']] = torch.inf
            ids = score.argmin(-1)
            paths.append(ids)
            recon += book[ids]; residual -= book[ids]
        results.append(decode(recon, config).cpu().numpy())
        codes_out.append(torch.stack(paths, -1).cpu().numpy().astype(np.int16))
    return np.concatenate(results), np.concatenate(codes_out)


def self_check():
    x = torch.tensor([[[1., 0., torch.pi / 2], [1., 1., torch.pi], [0., 1., -torch.pi / 2]]])
    assert torch.allclose(compose(to_diff(x)), x, atol=1e-6)
    for rep in ('diff', 'abs', 'sincos'):
        cfg = {'horizon': 3, 'representation': rep, 'scale': [1.] * (12 if rep == 'sincos' else 9)}
        reconstructed = decode(features(x, cfg), cfg)
        assert torch.allclose(reconstructed[..., :2], x[..., :2], atol=1e-6)
        assert wrap(reconstructed[..., 2] - x[..., 2]).abs().max() < 1e-6
    cfg = {'horizon': 1, 'representation': 'diff', 'scale': [1., 1., 1.]}
    books = torch.tensor([[[0., 0., 0.], [1., 0., .1]], [[0., 0., 0.], [.2, 0., .02]]])
    target = torch.tensor([[[1.2, 0., .12]], [[0., 0., 0.]]])
    decoded, codes = encode(target, torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, dtype=torch.long),
                            books, cfg, {'mean_yaw': 1}, beam_width=2, refine=1)
    assert np.allclose(decoded, target.numpy(), atol=1e-6)
    assert codes.tolist() == [[1, 1], [0, 0]]


if __name__ == '__main__':
    self_check()
    print('rvq self-check passed')
