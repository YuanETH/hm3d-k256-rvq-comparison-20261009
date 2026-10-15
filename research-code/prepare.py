"""Prepare geometry-only, scene-disjoint RVQ research caches. Never opens RGB."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

SEED = 'hm3d-rvq-research-20261015-v1'
OLD_SEED = 'hm3d-k256-rvq-paired-20261009-v1'


def hashed(value, seed=SEED):
    return hashlib.sha256((seed + ':' + value).encode()).hexdigest()


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def local_pose(world, base):
    delta = world[..., :2] - base[..., None, :2]
    c, s = np.cos(base[..., 2, None]), np.sin(base[..., 2, None])
    return np.stack([c * delta[..., 0] + s * delta[..., 1],
                     -s * delta[..., 0] + c * delta[..., 1],
                     wrap(world[..., 2] - base[..., 2, None])], axis=-1)


def fixed_windows(gt, horizon=10, dt=.5):
    count = len(gt) - 1
    grid = np.arange(count)[:, None] + np.arange(1, horizon + 1)[None] * dt / .5
    valid = grid <= count + 1e-8
    grid = np.minimum(grid, count)
    lo = np.floor(grid).astype(int)
    hi = np.minimum(lo + 1, count)
    frac = grid - lo
    world = gt[lo] + (gt[hi] - gt[lo]) * frac[..., None]
    world[..., 2] = gt[lo, 2] + wrap(gt[hi, 2] - gt[lo, 2]) * frac
    return local_pose(world, gt[:-1]).astype(np.float32), valid


def process_batch(job):
    batch, split, scratch_str, dataset_str, cache_str = job
    scratch, dataset, cache = map(Path, (scratch_str, dataset_str, cache_str))
    bid = batch['batch_id']
    out = cache / 'batches' / (bid + '.npz')
    if out.exists():
        with np.load(out) as x:
            return {'batch_id': bid, 'split': split, 'scene_id': batch['scene_id'],
                    'routes': len(x['offsets']) - 1, 'steps': len(x['truth']), 'cached': True}
    raw = (scratch / 'generated/chunks' / (bid + '.json')).read_bytes()
    receipt = json.loads((scratch / 'generated/receipts' / (bid + '.json')).read_text())
    sha = hashlib.sha256(raw).hexdigest()
    if sha != receipt['annotations_sha256']:
        raise ValueError('annotation receipt mismatch ' + bid)
    chunk = json.loads(raw)
    del raw
    eps = chunk['episodes']
    if len(eps) != batch['trajectory_count'] or sum(len(e['frames']) for e in eps) != batch['frame_count']:
        raise ValueError('roster mismatch ' + bid)
    wanted = {e['episode_id']: e for e in chunk['sampling']['selected']}
    by_shard = defaultdict(set)
    for e in wanted.values():
        by_shard[e['shard_file']].add(e['episode_id'])
    sources = {}
    for shard, ids in by_shard.items():
        with gzip.open(dataset / shard, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['episode_id'] in ids:
                    sources[row['episode_id']] = row
    if sources.keys() != wanted.keys():
        raise ValueError('source identity mismatch ' + bid)
    centers = np.asarray(json.loads((dataset / 'codebook-k256.json').read_text())['centers_dx_dy_dtheta'])
    arrays = defaultdict(list)
    offsets = [0]
    episode_ids, starts = [], []
    for ep in eps:
        frames = ep['frames']
        source = sources[ep['episode_id']]['trajectory']
        gt4 = np.asarray([s['pose'] for s in source['steps']], np.float64)
        if ep['fps'] != 2 or not np.allclose(gt4, [f['pose_world'] for f in frames], atol=1e-6):
            raise ValueError('pose/fps mismatch ' + ep['episode_id'])
        if source['video']['sha256'] != ep['source_video_sha256']:
            raise ValueError('video identity mismatch')
        if any(f['frame_index'] != i or abs(f['t_sec'] - i * .5) > 1e-7 for i, f in enumerate(frames)):
            raise ValueError('frame time mismatch')
        gt = gt4[:, [0, 1, 3]]
        tokens = np.array([s['metadata']['planner_action_token_id'] for s in source['steps']])
        if tokens[-1] != 0 or np.any((tokens[:-1] < 1) | (tokens[:-1] > 256)):
            raise ValueError('K256 motion/STOP contract')
        truth = local_pose(gt[1:, None], gt[:-1])[:, 0]
        rvqs = [f['rvq'] for f in frames[:-1]]
        target = np.array([r['target_waypoints_local'] for r in rvqs], np.float32)
        decoded = np.array([r['decoded_waypoints_local'] for r in rvqs], np.float32)
        times = np.array([r['waypoint_t_sec'] for r in rvqs], np.float64)
        valid = np.array([r['future_valid_mask'] for r in rvqs], bool)
        matches = np.isclose(times, .5, atol=1e-7) & valid
        if not np.all(matches.sum(1) == 1):
            raise ValueError('0.5 s valid target not unique')
        action_index = matches.argmax(1)
        if not np.allclose(target[np.arange(len(truth)), action_index], truth, atol=1e-5):
            raise ValueError('RVQ target alignment mismatch')
        fixed, fvalid = fixed_windows(gt)
        quarter, qvalid = fixed_windows(gt, dt=.25)
        if not np.allclose(fixed[:, 0], truth, atol=1e-5):
            raise ValueError('fixed-window target mismatch')
        for key, value in {
            'truth': truth, 'gt_next': gt[1:], 'ksteps': centers[tokens[:-1] - 1],
            'original_target': target, 'original_decoded': decoded, 'original_times': times,
            'original_valid': valid, 'original_codes': np.array([r['code_indices'] for r in rvqs], np.int16),
            'action_index': action_index.astype(np.int8), 'fixed': fixed, 'fixed_valid': fvalid,
            'quarter': quarter, 'quarter_valid': qvalid,
        }.items():
            if not np.isfinite(value).all():
                raise ValueError('nonfinite ' + key)
            arrays[key].append(value)
        episode_ids.append(ep['episode_id'])
        starts.append(gt[0])
        offsets.append(offsets[-1] + len(truth))
    payload = {k: np.concatenate(v) for k, v in arrays.items()}
    payload.update(offsets=np.array(offsets), starts=np.array(starts), episode_ids=np.array(episode_ids),
                   annotation_sha256=np.array(sha), scene_id=np.array(batch['scene_id']))
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, **payload)
    tmp.replace(out)
    return {'batch_id': bid, 'split': split, 'scene_id': batch['scene_id'],
            'routes': len(eps), 'steps': offsets[-1], 'annotation_sha256': sha}


def combine(cache, split, roster):
    parts = []
    offsets = [0]
    starts, episodes, route_scenes, action_scenes = [], [], [], []
    fields = defaultdict(list)
    for scene_index, b in enumerate(roster):
        with np.load(cache / 'batches' / (b['batch_id'] + '.npz')) as x:
            for name in ('truth', 'gt_next', 'ksteps', 'original_target', 'original_decoded',
                         'original_times', 'original_valid', 'original_codes', 'action_index',
                         'fixed', 'fixed_valid', 'quarter', 'quarter_valid'):
                fields[name].append(x[name])
            offsets.extend((x['offsets'][1:] + offsets[-1]).tolist())
            starts.append(x['starts'])
            episodes.append(x['episode_ids'])
            route_scenes.extend([scene_index] * len(x['starts']))
            action_scenes.extend([scene_index] * len(x['truth']))
    payload = {k: np.concatenate(v) for k, v in fields.items()}
    payload.update(offsets=np.array(offsets), starts=np.concatenate(starts), episode_ids=np.concatenate(episodes),
                   route_scenes=np.array(route_scenes), action_scenes=np.array(action_scenes),
                   scenes=np.array([b['scene_id'] for b in roster]))
    np.savez(cache / (split + '.npz'), **payload)
    return {'scenes': len(roster), 'routes': len(starts) and len(payload['starts']),
            'steps': len(payload['truth']), 'max_original_action_index': int(payload['action_index'].max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / 'batches').mkdir(exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.scratch / 'generation_units_20261005/generation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    by_scene = defaultdict(list)
    for b in manifest['batches']:
        by_scene[b['scene_id']].append(b)
    historical = set(sorted(by_scene, key=lambda s: hashed(s, OLD_SEED))[:60])
    fresh = sorted(set(by_scene) - historical, key=hashed)
    groups = {'train': fresh[:64], 'validation': fresh[64:76], 'test': fresh[76:100]}
    assert not any(set(a) & set(b) for i, a in enumerate(groups.values()) for b in list(groups.values())[i + 1:])
    roster = {name: [min(by_scene[s], key=lambda b: hashed(b['batch_id'])) for s in scenes]
              for name, scenes in groups.items()}
    private = {'seed': SEED, 'groups': groups, 'batches': roster}
    (args.out / 'private_splits.json').write_text(json.dumps(private, indent=2) + '\n')
    protocol = {'version': 1, 'seed': SEED, 'generation_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'private_split_sha256': hashlib.sha256(json.dumps(private, sort_keys=True).encode()).hexdigest(),
                'scene_split': {k: len(v) for k, v in groups.items()}, 'historical_scenes_excluded': len(historical),
                'training_window_limit_per_scene': 2048, 'primary_metric': 'route-equal mean full-route endpoint error',
                'selection': 'validation only; freeze variants and model hashes before test evaluation',
                'source_dt_s': .5, 'video_reads': False,
                'scope': 'offline quantization and trajectory reconstruction, no visual policy or robot controller'}
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    jobs = [(b, name, str(args.scratch), str(args.dataset), str(args.cache)) for name, rows in roster.items() for b in rows]
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_batch, j) for j in jobs]
        for f in as_completed(futures):
            result = f.result(); receipts.append(result)
            print(f'{len(receipts)}/{len(jobs)} {result["split"]} {result["routes"]} routes {result["steps"]} steps', flush=True)
    counts = {name: combine(args.cache, name, rows) for name, rows in roster.items()}
    protocol['counts'] = counts
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.out / 'private_cache_receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(json.dumps(counts), flush=True)


def self_check():
    gt = np.array([[0., 0., 0.], [1., 0., np.pi / 2], [1., 1., np.pi]])
    wp, mask = fixed_windows(gt, 3)
    assert np.allclose(wp[1, 0], [1., 0., np.pi / 2])
    assert np.allclose(wp[0, 1, :2], [1., 1.]) and not mask[0, 2]
    q, _ = fixed_windows(gt, 4, .25)
    assert np.allclose(q[0, 1], wp[0, 0])
    gt = np.array([[0., 0., np.deg2rad(179)], [0., 0., np.deg2rad(-179)]])
    q, _ = fixed_windows(gt, 2, .25)
    assert np.isclose(q[0, 0, 2], np.deg2rad(1))


if __name__ == '__main__':
    self_check()
    main()
