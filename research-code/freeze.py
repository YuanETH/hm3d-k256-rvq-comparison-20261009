"""Freeze validation choices and model hashes before opening test scores."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path('results')
path = root / 'selection_lock.json'
if path.exists():
    raise RuntimeError('selection is already frozen; retain the existing research record')
definitions = json.loads((root / 'experiments.json').read_text())
metrics = {name: json.loads((root / 'validation' / name / 'metrics.json').read_text())
           for name in definitions['variants']}
eligible = [name for name, spec in definitions['variants'].items()
            if spec['horizon'] == 5 and spec['sampling'] == 'fixed' and spec['tokens'] == 3
            and spec['family'] == 'new_codebook' and spec.get('objective', {}).get('ade', 1) > 0]
preferred = min(eligible, key=lambda name: metrics[name]['route_endpoint_m']['mean'])
lock = {'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'variant_grid_sha256': hashlib.sha256((root / 'experiments.json').read_bytes()).hexdigest(),
    'variants': definitions['variants'],
    'model_sha256': {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'models').glob('*.npz')},
    'selected_on_validation': {'preferred_5point_3token_full_window': preferred,
        'reference_5point_diff_3seed': 'new_h5_diff_y1_beam', 'single_action_precision': 'new_h1_diff_y1_beam',
        'five_point_control_only': 'new_h5_action_only', 'four_token_5point': 'new_h5_4token'},
    'selection_rule': 'Fixed 0.5 s, three motion tokens, H5 with positive full-window ADE weight; minimize validation mean route endpoint error. All variants frozen before test evaluation.'}
path.write_text(json.dumps(lock, indent=2) + '\n')
print(preferred)
