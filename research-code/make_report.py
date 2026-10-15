"""Publish aggregate RVQ research, scientific figures, and anonymized overlays."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from experiment import integrate
from prepare import hashed

BLUE, GREEN, RED = '#1676be', '#159b78', '#df5b4c'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})


def figure(path, names, values, labels, ylabel, title, colors=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(np.arange(len(names)), values, color=colors or BLUE, width=.65)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(len(names)), labels, rotation=12)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.23)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def overlay(path, gt, k, r):
    w, h, pad = 760, 590, 68
    all_xy = np.vstack([gt, k, r])[:, :2]
    middle = (all_xy.max(0) + all_xy.min(0)) / 2
    span = max(np.ptp(all_xy, axis=0).max(), 3.) * 1.16
    scale = min((w - 2 * pad) / span, (h - 2 * pad) / span)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 17)
    small = ImageFont.truetype(font_path, 14)
    def point(x):
        return int(w / 2 + (x[0] - middle[0]) * scale), int(h / 2 - (x[1] - middle[1]) * scale)
    images = []
    for end in np.unique(np.linspace(0, len(gt) - 1, min(75, len(gt)), dtype=int)):
        image = Image.new('RGB', (w, h), '#f7f9fc'); d = ImageDraw.Draw(image)
        for x in range(pad, w - pad, 55): d.line((x, pad, x, h - pad), fill='#e3e9f0')
        for y in range(pad, h - pad, 55): d.line((pad, y, w - pad, y), fill='#e3e9f0')
        d.text((pad, 18), f'Top-down NWU metres | t = {end * .5:.1f} s | arrows = heading', fill='#172033', font=font)
        scale_m = min((.1, .2, .5, 1, 2, 5, 10, 20, 50), key=lambda m: abs(m * scale - 90))
        sx, sy = w - pad, 54
        d.line((sx - scale_m * scale, sy, sx, sy), fill='#49586b', width=2)
        d.line((sx, sy - 4, sx, sy + 4), fill='#49586b', width=2)
        d.line((sx - scale_m * scale, sy - 4, sx - scale_m * scale, sy + 4), fill='#49586b', width=2)
        d.text((sx - scale_m * scale, sy - 19), f'{scale_m:g} m', fill='#49586b', font=small)
        d.line([point(x) for x in gt], fill='#c3cbd7', width=2)
        gx, gy = point(gt[-1]); d.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline='#242b3b', width=2)
        for j, (arr, color, label) in enumerate([(gt, '#242b3b', 'GT'), (k, RED, 'K256'), (r, GREEN, 'RVQ H5 improved')]):
            points = [point(x) for x in arr[:end + 1]]
            if len(points) > 1: d.line(points, fill=color, width=4, joint='curve')
            x, y = points[-1]; d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline='white')
            angle = arr[end, 2]
            arrow = point(arr[end, :2] + np.array([np.cos(angle), np.sin(angle)]) * max(.35, span * .035))
            d.line((x, y, *arrow), fill=color, width=4)
            for turn in (-.5, .5):
                back = (int(arrow[0] - 10 * np.cos(angle + turn)), int(arrow[1] + 10 * np.sin(angle + turn)))
                d.line((*arrow, *back), fill=color, width=3)
            d.rectangle((pad + 215 * j, h - 38, pad + 215 * j + 13, h - 25), fill=color)
            d.text((pad + 215 * j + 20, h - 42), label, font=small, fill='#172033')
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=130, loop=0, optimize=True)


def gallery(cache, root, assets, preferred):
    data = dict(np.load(cache / 'test.npz'))
    with np.load(root / 'test' / 'k256' / 'private_predictions.npz') as x: kstep, krow = x['steps'], x['routes']
    with np.load(root / 'test' / preferred / 'private_predictions.npz') as x: rstep, rrow = x['steps'], x['routes']
    delta = rrow[:, 0] - krow[:, 0]
    category = np.where(delta < -.05, 'rvq_better', np.where(delta > .05, 'k256_better', 'close'))
    used, cards = set(), []
    for kind in ('rvq_better', 'close', 'k256_better'):
        for low, high in ((5, 10), (10, 25)):
            candidates = np.flatnonzero((category == kind) & (krow[:, 3] >= low) & (krow[:, 3] < high) & (krow[:, 4] < 140))
            if not len(candidates): raise ValueError('no representative overlay candidate')
            median = np.median(delta[candidates])
            ordered = sorted(candidates, key=lambda i: (int(data['route_scenes'][i]) in used, abs(delta[i] - median), hashed(str(data['episode_ids'][i]))))
            i = ordered[0]; used.add(int(data['route_scenes'][i]))
            lo, hi = data['offsets'][i:i + 2]
            start = data['starts'][i]
            gt = np.vstack([start, data['gt_next'][lo:hi]])
            k = np.vstack([start, integrate(start, kstep[lo:hi])]); r = np.vstack([start, integrate(start, rstep[lo:hi])])
            name = f'research-route-{len(cards) + 1:02d}.gif'
            overlay(assets / name, gt, k, r)
            cards.append({'asset': name, 'category': kind, 'gt_length_m': float(krow[i, 3]), 'frames': len(gt),
                'k256_endpoint_m': float(krow[i, 0]), 'rvq_endpoint_m': float(rrow[i, 0]),
                'k256_ade_m': float(krow[i, 1]), 'rvq_ade_m': float(rrow[i, 1]),
                'gt_waypoints': len(gt), 'k256_waypoints': len(k), 'rvq_waypoints': len(r)})
    return cards, {kind: int(np.sum(category == kind)) for kind in ('rvq_better', 'close', 'k256_better')}


CSS = '''
:root{--ink:#172033;--muted:#58677b;--line:#dfe7ef;--blue:#1676be;--green:#159b78}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.7 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
header{background:#152d42;color:white;padding:48px 22px}.wrap,main{max-width:1150px;margin:auto}main{padding:0 22px 60px}
h1{font-size:clamp(29px,4vw,46px);line-height:1.2;margin:12px 0}h2{font-size:25px;margin:0 0 12px}h3{font-size:19px;margin:12px 0 6px}p{margin:8px 0 14px}
header p{color:#d8e8f3;max-width:950px}a{color:#075eaa}header a{color:#b8e8ff}code{background:#eef3f8;padding:1px 4px;border-radius:4px;overflow-wrap:anywhere}
section,.card{background:white;border:1px solid var(--line);border-radius:14px;padding:25px;margin:19px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{margin:18px 0 0;padding:18px}.card strong{font-size:29px;display:block}
.callout{border-left:5px solid var(--green);background:#edf9f4;padding:15px 18px;border-radius:7px;margin:16px 0}.note{border-left:4px solid #dca336;background:#fff8e9;padding:14px 18px;margin:16px 0;border-radius:7px}.muted,figcaption,small{color:var(--muted)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}figure img{width:100%;display:block}figcaption{padding:9px 12px;font-size:13px}
.scroll{overflow:auto}table{border-collapse:collapse;min-width:780px;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}thead{background:#edf3f8}td.note-col{text-align:left;min-width:150px}tbody tr:hover{background:#f5f9fc}.selected{background:#eff9f5}
.route{display:grid;grid-template-columns:.9fr 1.3fr;gap:16px;align-items:center;margin:14px 0;padding:13px;border:1px solid var(--line);border-radius:10px}.route img{width:100%}.tag{display:inline-block;background:#e9f6f1;color:#14634e;padding:3px 10px;border-radius:20px;font-size:13px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}select,input{font:inherit;padding:6px;border:1px solid #b8c7d7;border-radius:5px}details{margin:15px 0}summary{cursor:pointer;font-weight:650}ul{padding-left:24px}li{margin:7px 0}.eyebrow{letter-spacing:.1em;color:#a9e8ce;font-size:13px}
@media(max-width:760px){.grid,.cards,.route{grid-template-columns:1fr}section{padding:18px}.card{margin-top:8px}table{font-size:13px}}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--site', type=Path, required=True)
    args = p.parse_args()
    root, site = args.results, args.site
    s = json.loads((root / 'research_summary.json').read_text())
    variants = s['variants']
    preferred = s['selection']['selected_on_validation']['preferred_5point_3token_full_window']
    def metric(name, key='route_endpoint_m', split='test'): return variants[name][split][key]['mean']
    def test(name): return variants[name]['test']
    assets = site / 'assets'; assets.mkdir(exist_ok=True, parents=True)
    model_dir = site / 'research-models'; model_dir.mkdir(exist_ok=True)
    code_dir = site / 'research-code'; code_dir.mkdir(exist_ok=True)
    for file in Path(__file__).parent.glob('*.py'): shutil.copy2(file, code_dir / file.name)
    for file in (root / 'models').glob('*.npz'): shutil.copy2(file, model_dir / file.name)
    for name in ('protocol.json', 'experiments.json', 'selection_lock.json', 'exact_search_probe.json'):
        shutil.copy2(root / name, site / ('research-' + name))
    shutil.copy2(root / 'research_summary.json', site / 'research-summary.json')
    contract = {'variant': preferred, 'model_file': 'research-models/' + test(preferred)['config']['model'] + '.npz',
        'model_sha256': test(preferred)['model_sha256'], 'horizon': 5, 'motion_tokens_per_decision': 3,
        'levels': [256, 256, 256], 'waypoint_dt_s': .5, 'replan_dt_s': .5,
        'coordinate_frame': 'current body: x forward metres, y left metres, CCW yaw radians',
        'representation': 'absolute local SE2 poses; three codewords sum in scaled feature space',
        'execution': 'execute the first waypoint for one 0.5-second control period and replan; other four rows are lookahead',
        'terminal': 'motion codec has no reserved STOP index; use an explicit goal-stop flag or Affordance STOP token, and halt before applying decoded motion',
        'policy_requirement': 'relabel targets and fine-tune/train the policy using this exact codebook; never substitute these assets into an unchanged official checkpoint',
        'validated_scope': 'offline quantization/reconstruction only; controller and visual policy untested'}
    (site / 'research-deployment-contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    main_names = ['official_release', 'k256', 'new_h10_diff_y1_beam', preferred, 'new_h3_diff_y1_beam', 'new_h1_diff_y1_beam']
    main_labels = ['Released RVQ H10', 'K256', 'New RVQ H10 diff', 'Selected RVQ H5', 'New RVQ H3', 'New RVQ H1']
    figure(assets / 'research-overview.png', main_names, [metric(n) for n in main_names], main_labels,
           'Mean full-route endpoint error (m)', 'Independent test: identical source trajectories', [BLUE, RED, '#6d89bb', GREEN, '#52bda2', '#a4d9c4'])
    zoom = main_names[1:]
    figure(assets / 'research-precision.png', zoom, [metric(n) for n in zoom], main_labels[1:],
           'Mean full-route endpoint error (m)', 'Precision with deterministic waypoint times', [RED, '#6d89bb', GREEN, '#52bda2', '#a4d9c4'])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (name, color, label) in enumerate([('official_release', BLUE, 'Released RVQ'), ('k256', RED, 'K256'), (preferred, GREEN, 'Selected RVQ H5')]):
        values = [metric(name, key) for key in ('segment5_endpoint_m', 'segment10_endpoint_m', 'route_endpoint_m')]
        bars = ax.bar(np.arange(3) + (j - 1) * .26, values, .25, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(3), ['5 s re-anchored', '10 s re-anchored', 'Full route, start only'])
    ax.set_ylabel('Mean endpoint error (m)'); ax.set_title('Drift at multiple open-loop durations', fontweight='bold')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    fig.tight_layout(); fig.savefig(assets / 'research-open-loop.png', dpi=170); plt.close(fig)
    horizons = [10, 5, 3, 1]
    horizon_names = [f'new_h{h}_diff_y1_beam' for h in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, unit in [(axes[0], 'route_endpoint_m', 'Mean route endpoint error (m)'), (axes[1], 'step_yaw_deg', 'Mean 0.5 s yaw error (deg)')]:
        values = [metric(n, key) for n in horizon_names]
        bars = ax.bar([f'H={h}' for h in horizons], values, color=GREEN)
        ax.bar_label(bars, fmt='%.3f', padding=3); ax.set_ylabel(unit); ax.set_ylim(0, max(values) * 1.24)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Horizon ablation: same 3 x 256 RVQ, same diff representation, same loss', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-horizon.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, unit in [(axes[0], 'route_endpoint_m', 'Mean route endpoint error (m)'), (axes[1], 'step_yaw_deg', 'Mean 0.5 s yaw error (deg)')]:
        x = np.arange(3)
        for offset, encoder, color, label in [(-.18, 'l2', '#9fb3ca', 'Feature L2 greedy'), (.18, 'beam', GREEN, 'Geometry + execution yaw')]:
            values = [metric(f'new_h5_diff_y{weight}_{encoder}', key) for weight in ('0.3', '1', '3')]
            bars = ax.bar(x + offset, values, .36, color=color, label=label); ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        ax.set_xticks(x, ['0.3', '1.0', '3.0']); ax.set_xlabel('Heading feature scale (m/rad)'); ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.suptitle('H5 angle precision: training scale and assignment objective', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-yaw.png', dpi=170); plt.close(fig)
    loss_names = ['official_release', 'official_mask', 'official_search64', 'official_yaw_all10', 'official_exec30', 'official_prefix5']
    figure(assets / 'research-objective.png', loss_names, [metric(n) for n in loss_names],
        ['Released', 'Mask padding', 'Beam64 + refine', 'All-row yaw', 'Execution yaw', '5-row prefix'],
        'Mean route endpoint error (m)', 'Same official codebook; original GT-time sidecars', [BLUE] * 4 + [GREEN] * 2)
    rep_names = ['new_h5_diff_y1_beam', 'new_h5_abs_y1_beam', 'new_h5_sincos_y1_beam', 'factorized_xy2_yaw1', 'factorized_xy1_yaw2']
    figure(assets / 'research-representation.png', rep_names, [metric(n) for n in rep_names],
        ['SE2 diff', 'Ego absolute', 'Ego sin/cos', '2 XY + 1 yaw', '1 XY + 2 yaw'],
        'Mean route endpoint error (m)', 'H5 representation and token allocation: 3 tokens each')
    turn_keys = ['turn_lt1', 'turn_1to5', 'turn_5to15', 'turn_ge15']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, names, title in [(axes[0], ['official_release', 'k256', preferred], 'Mean absolute yaw error'),
                             (axes[1], ['k256', preferred], 'Zoom: mean absolute yaw error')]:
        width = .75 / len(names)
        for j, name in enumerate(names):
            values = [test(name)[key]['yaw_deg']['mean'] for key in turn_keys]
            color = BLUE if name == 'official_release' else RED if name == 'k256' else GREEN
            label = 'Released RVQ' if name == 'official_release' else 'K256' if name == 'k256' else 'Selected H5'
            bars = ax.bar(np.arange(4) + (j - (len(names) - 1) / 2) * width, values, width, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
        ax.set_xticks(range(4), ['<1°', '1–5°', '5–15°', '≥15°']); ax.set_xlabel('GT turn within 0.5 s')
        ax.set_ylabel('deg'); ax.set_title(title); ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout(); fig.savefig(assets / 'research-turn-strata.png', dpi=170); plt.close(fig)
    ex_names = ['official_release', preferred, 'new_h5_diff_y3_l2', 'new_h5_action_only']
    ex_labels = ['Released RVQ', 'Selected RVQ H5', 'H5 yaw3 greedy', 'H5 action-only']
    execution = s['execution']['test']
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for ax, key, title in [(axes[0], 'endpoint_m', 'Mean endpoint error (m)'), (axes[1], 'request_fraction_vs_k256', 'Requests / K256 requests')]:
        for j, (mode, color, label) in enumerate([('step', BLUE, 'Replan every 0.5 s'), ('up_to_1s', '#8ab9c3', 'Execute up to 1 s'), ('full_chunk', GREEN, 'Execute full valid chunk')]):
            values = [execution[n][mode][key]['mean'] if key == 'endpoint_m' else execution[n][mode][key] for n in ex_names]
            bars = ax.bar(np.arange(len(ex_names)) + (j - 1) * .26, values, .25, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=7)
        ax.set_xticks(range(len(ex_names)), ex_labels, rotation=15); ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Execution cadence changes the required quantization objective', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-execution.png', dpi=170); plt.close(fig)
    prefix = s['prefix_analysis']['test']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, ylabel in [(axes[0], 'local_ade_m', 'Mean H5 local ADE (m)'), (axes[1], 'route_endpoint_m', 'Mean route endpoint error (m)')]:
        values = [prefix[str(level)][key]['mean'] for level in (1, 2, 3)]
        bars = ax.bar(['L0', 'L0 + L1', 'L0 + L1 + L2'], values, color=['#b7d9ce', '#6ab69c', GREEN])
        ax.bar_label(bars, fmt='%.3f', padding=3); ax.set_ylabel(ylabel); ax.set_ylim(0, max(values) * 1.23)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Selected H5: decoding prefixes of the frozen three-token assignment', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-prefix.png', dpi=170); plt.close(fig)
    cards, categories = gallery(args.cache, root, assets, preferred)
    (site / 'research-gallery.json').write_text(json.dumps({'cases': cards, 'endpoint_gap_threshold_m': .05, 'counts': categories}, indent=2) + '\n')
    category_names = {'rvq_better': '5 点 RVQ 占优', 'close': '终点误差接近', 'k256_better': 'K256 占优'}
    gallery_html = ''.join(f'''<article class="route"><div><span class="tag">{category_names[g['category']]}</span>
<h3>轨迹 {i} · GT 路程 {g['gt_length_m']:.1f} m</h3><p>终点误差：K256 {g['k256_endpoint_m']:.3f} m / RVQ {g['rvq_endpoint_m']:.3f} m<br>整轨 ADE：K256 {g['k256_ade_m']:.3f} m / RVQ {g['rvq_ade_m']:.3f} m</p>
<p><strong>绘制路点：K256 {g['k256_waypoints']} 个；RVQ {g['rvq_waypoints']} 个</strong>，均含共同起点。GT {g['gt_waypoints']} 个。</p><small>RVQ 内部每次编码 5 个未来局部点，动图按每 0.5 秒执行首点积分。箭头显示朝向。</small></div><img loading="lazy" src="assets/{g['asset']}" alt="GT、K256 和优选五点 RVQ 的同图俯视轨迹动图，含朝向箭头"></article>''' for i, g in enumerate(cards, 1))
    preferred_test = test(preferred); original = test('official_release'); k = test('k256')
    reduction_old = 1 - metric(preferred) / metric('official_release')
    reduction_k = 1 - metric(preferred) / metric('k256')
    ci = preferred_test['paired']['k256']['scene_95ci_delta_m']
    def row(name, label):
        v = test(name)
        return f'''<tr class="{'selected' if name == preferred else ''}"><th>{label}</th><td>{v['config']['tokens']}</td><td>{v['config']['horizon']}</td><td>{metric(name):.4f}</td><td>{v['route_endpoint_m']['p95']:.4f}</td><td>{v['step_yaw_deg']['mean']:.4f}</td><td>{v['local_ade_m']['mean']:.4f}</td></tr>'''
    main_rows = ''.join(row(name, label) for name, label in [('official_release', '原发版 RVQ · 几何取点'), ('k256', '原 K256'), ('new_h10_diff_y1_beam', '新 RVQ · 固定时间 10 点'), ('new_h5_diff_y1_beam', '新 RVQ · 固定时间 5 点 / 差分'), (preferred, '验证集优选 · 固定时间 5 点 / 绝对坐标'), ('new_h5_control_weighted', '候选 · 5 点差分 / 额外首点位置损失'), ('new_h3_diff_y1_beam', '新 RVQ · 固定时间 3 点'), ('new_h1_diff_y1_beam', '新 RVQ · 固定时间 1 点'), ('new_h5_action_only', '5 点码本 · 仅优化首步'), ('new_h5_4token', '5 点 RVQ · 4 token')])
    family_labels = {'baseline': 'K256 基准', 'control': '无量化控制', 'official': '固定官方码本', 'new_codebook': '新联合 RVQ', 'factorized': '分离平移/角度', 'hybrid': '额外角度 token'}
    all_rows = []
    for name, record in variants.items():
        v, val = record['test'], record['validation']; cfg = v['config']
        annotations = []
        if cfg['sampling'] == 'original': annotations.append('依赖 GT 时间侧车')
        if cfg.get('objective', {}).get('ade', 1) == 0: annotations.append('仅优化执行点')
        if cfg.get('tokens', 0) > 3: annotations.append('增加 token')
        if name == preferred: annotations.append('验证集优选')
        if cfg.get('model', '').endswith(('seed2', 'seed3')): annotations.append('随机种子重复')
        all_rows.append(f'''<tr data-family="{cfg['family']}" data-tokens="{cfg['tokens']}" data-name="{name}" class="{'selected' if name == preferred else ''}"><td><code>{name}</code></td><td>{cfg['tokens']}</td><td>{cfg['horizon']}</td><td>{val['route_endpoint_m']['mean']:.4f}</td><td>{v['route_endpoint_m']['mean']:.4f}</td><td>{v['route_endpoint_m']['p95']:.4f}</td><td>{v['step_yaw_deg']['mean']:.4f}</td><td>{v['step_xy_m']['mean']:.4f}</td><td>{v['local_ade_m']['mean']:.4f}</td><td class="note-col">{'；'.join(annotations) or family_labels[cfg['family']]}</td></tr>''')
    all_rows = ''.join(all_rows)
    option_html = ''.join(f'<option value="{key}">{value}</option>' for key, value in family_labels.items())
    horizon_rows = ''.join(f'''<tr><th>{h} 点</th><td>{metric(f'new_h{h}_diff_y1_beam'):.4f}</td><td>{metric(f'new_h{h}_diff_y1_beam','step_yaw_deg'):.4f}</td><td>{test(f'new_h{h}_diff_y1_beam')['window_travel_m']['mean']:.3f}</td><td>{h * .5:.1f}</td><td>{'—' if h < 5 else format(test(f'new_h{h}_diff_y1_beam')['shared_first5_ade_m']['mean'], '.4f')}</td></tr>''' for h in horizons)
    exact = s['exact_search_probe']['models']
    exact_rows = ''.join(f'''<tr><th>{'原编码器' if name == 'official_reproduce' else '优选 5 点编码器'}</th><td>{value['candidate_tuples_per_window']:,}</td><td>{value['within_1e_minus5_fraction']:.2%}</td><td>{value['objective_gap']['mean']:.6f}</td></tr>''' for name, value in exact.items())
    seed_names = ['new_h5_diff_y1_beam', 'new_h5_diff_y1_seed2_beam', 'new_h5_diff_y1_seed3_beam']
    seed_endpoints = [metric(n) for n in seed_names]
    category_rows = ''.join(f'<tr><th>{category_names[name]}</th><td>{count:,}</td><td>{count / preferred_test["routes"]:.2%}</td></tr>' for name, count in categories.items())
    execution_rows = ''.join(f'''<tr><th>{name}</th><td>{value['step']['endpoint_m']['mean']:.4f}</td><td>{value['up_to_1s']['endpoint_m']['mean']:.4f}</td><td>{value['full_chunk']['endpoint_m']['mean']:.4f}</td><td>{value['full_chunk']['mean_execution_s']:.2f}</td><td>{value['full_chunk']['request_fraction_vs_k256']:.1%}</td></tr>''' for name, value in ((n, execution[n]) for n in ex_names))
    summary_url = 'research-summary.json'
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RVQ 开环精度研究 · HM3D</title><meta name="description" content="60 个 RVQ 变体、场景独立验证与测试、五路点编码、角度精度和执行频率研究。"><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="eyebrow">RVQ TRAJECTORY QUANTIZATION STUDY · 2026-10-15</div><h1>RVQ 如何降低开环误差</h1><p>同源 HM3D 轨迹 · 60 个编码变体 · 16 份训练码本 · 64 / 12 / 24 场景互斥划分。研究编码精度、未来轨迹质量与执行方式之间的关系。</p><a href="index.html">原 K256 / RVQ 对照报告</a> · <a href="{summary_url}">完整机器可读结果</a> · <a href="research-bundle.zip">研究代码与码本下载</a></div></header><main>
<div class="cards"><div class="card"><small>独立测试</small><strong>{preferred_test['routes']:,} 条轨迹</strong><small>{preferred_test['scenes']} 个场景 · {preferred_test['steps']:,} 个动作</small></div><div class="card"><small>优选 5 点 RVQ 整轨终点误差</small><strong>{metric(preferred):.3f} m</strong><small>3 个动作 token · 每 0.5 秒重新规划</small></div><div class="card"><small>优选方案的单步角度误差</small><strong>{preferred_test['step_yaw_deg']['mean']:.3f}°</strong><small>原发版 RVQ 为 {original['step_yaw_deg']['mean']:.3f}°</small></div></div>
<section><h2>在保持未来轨迹的同时，5 点 RVQ 可以显著降低漂移</h2><div class="callout">按验证集事先选定的 5 点、3 token 方案，在新测试集的整轨终点误差均值为 <strong>{metric(preferred):.4f} m</strong>，相比原发版 RVQ 的 {metric('official_release'):.4f} m 降低 {reduction_old:.1%}，相比同批 K256 的 {metric('k256'):.4f} m 降低 {reduction_k:.1%}。RVQ−K256 的配对均值差 95% 场景 bootstrap 区间为 [{ci[0]:.4f}, {ci[1]:.4f}] m。</div>
<p>该方案使用固定 0.5 秒时间网格、5 个未来局部 SE(2) 位姿、三层各 256 项的 RVQ。码本在训练场景上拟合；编码时同时优化整个有效窗口的位置、各点方向与本次执行点方向。窗口 ADE 为 {preferred_test['local_ade_m']['mean']:.4f} m，P95 为 {preferred_test['local_ade_m']['p95']:.4f} m，保留未来几何信息。预算是每次 3 个动作 token，K256 为 1 个；两者提供的信息量不同。优选来自验证集，测试集打开前已冻结所有变体与码本哈希。</p>
<div class="grid"><figure><img src="assets/research-overview.png" alt="原RVQ、K256与不同路点数新RVQ的终点误差柱状图"><figcaption>相同的 3,760 条测试轨迹；所有终点误差均从共同 GT 起点开环积分。</figcaption></figure><figure><img src="assets/research-precision.png" alt="固定时间RVQ与K256精度对比柱状图"><figcaption>放大低误差区间。1 点方案只保留下一步，不能据此宣称 5 点或 10 点未来预测同样精确。</figcaption></figure></div>
<div class="scroll"><table><thead><tr><th>用法</th><th>动作 token</th><th>内部路点</th><th>整轨终点均值 m</th><th>终点 P95 m</th><th>单步方向 °</th><th>自身窗口 ADE m</th></tr></thead><tbody>{main_rows}</tbody></table></div>
<p class="muted">不同 H 的“自身窗口 ADE”覆盖范围不同；比较 5 点与 10 点时，下方另给相同前 5 点 ADE。表中 token 数只计算动作；Pointing 的 Object / Affordance 仍另占 2 个 token。</p></section>
<section><h2>另一个低误差候选：同时加强首点位置与方向</h2><p><code>new_h5_control_weighted</code> 在 H5 差分损失中额外加入首点位置误差，仍保留整个窗口 ADE。其验证集终点均值为 {metric('new_h5_control_weighted', split='validation'):.4f} m，测试为 {metric('new_h5_control_weighted'):.4f} m，局部窗口 ADE 为 {metric('new_h5_control_weighted','local_ade_m'):.4f} m；三个种子的测试结果为 {metric('new_h5_control_weighted'):.4f}、{metric('new_h5_control_seed2'):.4f}、{metric('new_h5_control_seed3'):.4f} m。这是一种保留前瞻且开环误差小的候选用法。</p><p>它在验证集上的排序略落后于绝对坐标 H5，因此保留测试前冻结的优选标记。报告同时展示其更低的测试观察值，供后续在实际任务验证集上比较；没有根据本次测试排名重新挑选种子或更新码本。</p></section>
<section><h2>先排除协议、时间和实现因素</h2><ul>
<li>每条轨迹核对源视频身份、帧数、GT 位姿、2 Hz 时间轴和 K256 STOP 约定；每个 batch 校验原标注 SHA 收据。只读几何，RGB 读取为 0。</li>
<li>测试保留全部非终止动作，没有按旧 RVQ 的 training_eligible 标签或量化误差筛选。所有变体使用相同的轨迹和动作。</li>
<li>原发版编码使用当前帧 GT 起点的重叠未来窗；3 个码字求和后解码，不是每个 token 各编码一段轨迹。原几何细分窗以距离≤0.25 m、转角≤30°取前 10 点，0.5 秒执行点的索引会随细分数变化。</li>
<li>新固定时间方案的第一个路点始终对应下一源帧，既不需要 GT 侧车时间戳，也不把不足 H 点的尾部填充纳入编码损失和指标。0.25 秒变体只插值原 2 Hz GT，没有增加真实观测。</li>
<li>无量化 GT 增量重建在测试集的最大终点误差为 {s['audits']['identity_max_endpoint_m']['test']:.2e} m。发版编码器在完整验证集重编码 token 一致率为 {s['audits']['validation_encoder_reproduction']['same_code_tuple_fraction']:.2%}；优选方案的独立 NumPy 解码器验证 {s['audits']['independent_numpy_decoder']['windows']:,} 个窗口，位置最大差 {s['audits']['independent_numpy_decoder']['max_xy_abs_difference_m']:.2e} m。</li>
</ul><p>编码目标由各帧 GT 未来窗口生成，回放时将解码动作应用在当前已重建位姿上，整轨仅在初始位姿对齐。整轨指标对轨迹等权；单步指标对动作等权，局部 ADE 先在有效点内取均值，再对窗口等权。5 秒与 10 秒实验分别每 10、20 个动作从 GT 起点重置，段内仍开环积分。</p>
<figure><img src="assets/research-open-loop.png" alt="5秒、10秒和整轨开环终点误差柱状图"><figcaption>不同开环时长均比较同一批可用段。完整段数和分位数记录在机器可读结果中。</figcaption></figure>
<p class="muted">测试集重编码有 {s['audits']['test_encoder_near_ties']['different_code_windows']} / {preferred_test['steps']:,} 个近似并列窗口选择了不同 token，损失差仅 {s['audits']['test_encoder_near_ties']['max_objective_difference']:.2e}，导致整轨终点均值差 {s['audits']['test_encoder_near_ties']['mean_route_endpoint_delta_m']:.2e} m。原发版 baseline 始终使用保存的原 token，未以重编码替代。</p><div class="note">本研究衡量 GT 轨迹经过量化后的重建误差。没有运行视觉策略、碰撞检测、机器人动力学或反馈控制，因此这些数字用于选择编码方案，不是导航成功率。原 K256 码本曾使用全保留集拟合，本次新测试场景对它不构成严格训练隔离；新增 RVQ 码本则只接触 64 个训练场景。表中的 <code>new_h1_1token</code> 提供了同训练样本的单层 VQ 对照。</div></section>
<section><h2>5 点与 10 点：控制信息预算和时间轴后比较</h2><p>本组统一采用 SE(2) 差分特征、角度尺度 1 m/rad、三层各 256 项、固定 0.5 秒路点间隔、宽度 8 的几何搜索，并使用同一组 {s['model_summary']['new_h5_diff_y1']['config']['training_windows']:,} 个训练锚点。总预算仍为 24 bit；缩短 H 会把更多表示能力留给每个路点。</p>
<figure><img src="assets/research-horizon.png" alt="同样3token的10、5、3、1路点RVQ误差柱状图"><figcaption>控制了码本大小、表示、角度尺度和分配目标；变化项为真实编码的路点数。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>路点数</th><th>整轨终点均值 m</th><th>单步方向误差 °</th><th>窗口 GT 平均路程 m</th><th>最大前瞻 s</th><th>相同前5点 ADE m</th></tr></thead><tbody>{horizon_rows}</tbody></table></div>
<p><strong>平均覆盖距离：</strong>优选绝对坐标 H5 窗口平均沿 GT 走 {preferred_test['window_travel_m']['mean']:.3f} m，原几何 H10 窗口为 {original['window_travel_m']['mean']:.3f} m。距离是当前帧至各有效未来点的水平路程之和，包含首段、忽略终点填充。H5 固定网格最多前瞻 2.5 秒，轨迹尾部不足时按有效点计算。</p>
<p>“只编码 5 点”在这里包含两种明确不同的实验：<code>official_prefix5</code>保留原 10 点码本的前三个 token，只优化并使用其前 5 点，仍依赖原 GT 时间侧车；<code>new_h5_*</code>则训练了真正 15 维或角度 sin/cos 20 维的 5 点 RVQ。直接把官方 H10 manifest 改成 H5 无法得到这样的新码本。</p></section>
<section><h2>角度精度取决于训练尺度，也取决于分配目标</h2><p>角度权重从 0.3、1 到 3 m/rad 分别训练；每份码本同时比较“特征空间贪心残差分配”和“解码后几何损失 + 执行点方向”的搜索。增加角度权重会改变平移与方向的容量分配；结果并非权重越大就所有指标越好。</p>
<p>RVQ 的核心是 <code>ẑ = C₀[k₀] + C₁[k₁] + C₂[k₂]</code>：先拟合第一层码本，再对第一层留下的残差拟合第二层，再拟合第三层残差。本研究用 residual k-means、每层 256 项、k-means++ 初始化与两次重启训练；改变的是特征表示、尺度和编码分配目标，三个 token 并不各自对应一段独立轨迹。</p>
<div class="grid"><figure><img src="assets/research-yaw.png" alt="RVQ角度权重和编码目标对方向及终点误差的影响"><figcaption>同一 H5 码本内对照分配目标，横向比较角度训练尺度。</figcaption></figure><figure><img src="assets/research-representation.png" alt="SE2差分、绝对坐标、sin cos以及分离角度码位的对比"><figcaption>sin/cos 避免角度坐标的 ±π 跳变；本批数据中它没有明显超过绝对角度表示。分开分配码位也没有自动获得最好结果。</figcaption></figure></div>
<p>优选方案将 <code>[x₁,y₁,θ₁,…,x₅,y₅,θ₅]</code> 联合 RVQ 编码，解码输出为当前机体坐标下的绝对局部位姿。编码分配损失为 <code>ADE_valid + 0.3 × mean|wrap(角度误差)| + 1.0 × |wrap(首点角度误差)|</code>；角度以 rad 计。该目标保留整个窗口位置精度，同时直接约束每 0.5 秒执行的方向。</p>
<figure><img src="assets/research-turn-strata.png" alt="按真实转向幅度分层的K256、原RVQ和优选RVQ角度误差柱状图"><figcaption>在相同 GT 转向组内比较，右图放大 K256 与优选方案，避免大误差柱掩盖细节。</figcaption></figure>
<p>对 {preferred_test['in_place_turn']['yaw_deg']['n']:,} 个原地转向动作（水平移动小于 2 cm、转角至少 5°），原 RVQ 的平均方向误差为 {original['in_place_turn']['yaw_deg']['mean']:.3f}°，K256 为 {k['in_place_turn']['yaw_deg']['mean']:.3f}°，优选 H5 为 {preferred_test['in_place_turn']['yaw_deg']['mean']:.3f}°，其 P95 为 {preferred_test['in_place_turn']['yaw_deg']['p95']:.3f}°。这些动作几乎没有 XY 位移，直接用轨迹切线替代显式角度会丢失转向信息。</p>
<p>另有“4 个联合残差 token”和“原 RVQ + 8 bit 角度修正 token”实验，全部显式计为 4 个动作 token。后者只修正被执行点的方向，没有凭空增加整个未来窗口的精度；完整表中保留它的误差、角度饱和数和额外成本。</p></section>
<section><h2>搜索得更彻底，不能弥补与执行方式不一致的目标</h2><figure><img src="assets/research-objective.png" alt="保持原码本时改变填充、搜索宽度与方向目标的开环结果"><figcaption>这组使用原几何采样和 GT 侧车时间戳，只用于隔离编码分配目标。它们不能直接当作固定时间部署方案。</figcaption></figure>
<p>原目标主要看窗口位置与末点方向。宽度 64 搜索和逐层细化更充分地优化该目标，却没有解决逐帧开环漂移；加入执行点方向、或优化前 5 点，改善更明显。对固定时间网格，直接复用原码本仍可能因运动范围和采样分布不同而出现较大位置误差，见 <code>official_fixed05*</code>。因此需要同时检查目标、采样与码本分布。</p>
<p>额外对 64 个验证窗口做完整三层组合穷举，以区分 RVQ 容量和近似搜索误差。搜索访问每个允许的组合；下表的“最优”只指该窗口的给定损失，容差为 10⁻⁵。</p>
<div class="scroll"><table><thead><tr><th>编码器</th><th>每窗口穷举组合数</th><th>近似搜索达到最优比例</th><th>平均损失差</th></tr></thead><tbody>{exact_rows}</tbody></table></div><p class="muted">损失含位置与加权角度，不能把损失差直接解释为纯位置误差。该小样本核查没有证明整个测试集或整条轨迹的全局最优。</p></section>
<section><h2>执行首点和执行整段，需要分别评估</h2><figure><img src="assets/research-execution.png" alt="同一码本逐帧、每一秒和整段执行的误差与请求数柱状图"><figcaption>三种执行方式使用相同发版/重编码 token，按共同 2 Hz 网格回放。整段的实际时长由有效路点决定。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>编码方案</th><th>逐帧终点 m</th><th>最多1秒终点 m</th><th>整段终点 m</th><th>整段平均执行 s</th><th>整段请求/K256</th></tr></thead><tbody>{execution_rows}</tbody></table></div>
<p>为首点精度优化的编码，后续方向未必同样准确；降低请求数时必须重新检查整轨漂移。只优化下一步的 H5，其逐帧结果为 {metric('new_h5_action_only'):.4f} m，但自身窗口 ADE 为 {metric('new_h5_action_only','local_ade_m'):.4f} m，直接整段执行的误差为 {execution['new_h5_action_only']['full_chunk']['endpoint_m']['mean']:.3f} m。它适合研究单步控制的精度上限，不能以“5 点”名称替代未来轨迹质量检查。完整执行结果同时记录速度 jerk、方向角加速度和请求数。</p></section>
<section><h2>残差层与随机种子</h2><figure><img src="assets/research-prefix.png" alt="优选RVQ只解码第一层、两层和三层时的误差柱状图"><figcaption>截取同一组三 token 分配的前缀，展示残差层作用；没有为短前缀重新选择最优 token。</figcaption></figure>
<p>对 H5 差分基线使用相同训练锚点、三个不同 k-means 随机种子，测试终点误差均值分别为 {', '.join(f'{x:.4f}' for x in seed_endpoints)} m，均值 {np.mean(seed_endpoints):.4f} m、样本标准差 {np.std(seed_endpoints, ddof=1):.4f} m。三者实际都使用前三层；首个文件另存的第 4 层是后续追加训练，不会回写前三层，仅供 4 token 变体使用。控制加权版本也保留三个种子结果。绝对坐标优选码本是预先冻结的单个模型；场景 bootstrap 区间描述这份模型的数据抽样不确定性，不代表全部训练随机性。</p></section>
<section><h2>优选 5 点 RVQ 与 K256 的同图俯视动图</h2><p>所有测试轨迹按 Δ＝RVQ终点误差−K256终点误差分为三组，展示阈值为 ±0.05 m。下面每组各选 2 条，不按总体比例分配。各组在 5–10 m 与 10–25 m 路程内分别选择终点差接近候选中位数、少于 140 帧且尽量不同场景的案例；不用极端胜例或极端失败例代表整体。</p>
<div class="scroll"><table><thead><tr><th>类别</th><th>轨迹数</th><th>测试占比</th></tr></thead><tbody>{category_rows}</tbody></table></div>
<p>黑色 GT、红色 K256、绿色优选 RVQ；三者共享坐标尺度，右上角有米制比例尺，GT 终点用空心圆标记。箭头表示当前朝向，原地转向也能观察。绘制路点包含起点，每个 0.5 秒动作增加一个点，终止 STOP 不增加点；并未把每帧 5 个重叠局部点直接拼成世界路径。长轨迹播放最多抽取 75 帧，最终曲线保留全部点。背景网格不表示 NavMesh 或墙体。</p>{gallery_html}</section>
<section><h2>全部 60 个编码变体</h2><p>每个变体都完成了相同的 1,992 条验证轨迹和 3,760 条测试轨迹。表格展示逐帧执行结果，保留未改善、增加 token、依赖 GT 时间侧车以及牺牲前瞻精度的实验。测试结果没有用于重新选择或训练码本。</p>
<div class="controls"><label>系列 <select id="family"><option value="">全部</option>{option_html}</select></label><label>动作 token <select id="tokens"><option value="">全部</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label><label>搜索 <input id="search" placeholder="H5 / yaw / official"></label></div><p id="visible" class="muted"></p>
<div class="scroll"><table id="all"><thead><tr><th>变体</th><th>token</th><th>H</th><th>验证终点均值 m</th><th>测试终点均值 m</th><th>测试终点P95 m</th><th>单步方向 °</th><th>单步位置 m</th><th>自身窗口 ADE m</th><th>说明</th></tr></thead><tbody>{all_rows}</tbody></table></div>
<p class="muted">各系列含不同信息预算与时间约定。不同 H 的窗口 ADE 不等覆盖范围；新增 token 方案不作为同预算胜例。完整置信区间、转向分层、首5点同时间 ADE、方向残差相关性、码本哈希和执行结果见 <a href="research-summary.json">research-summary.json</a>。</p></section>
<section><h2>使用建议与可复现材料</h2><div class="callout">保留 5 点未来信息时，优先采用本次验证集选出的固定 0.5 秒 H5、3 层 RVQ 与执行点方向损失，并逐帧执行首点。若只需要下一动作，可选 H1 的 2 或 3 层 RVQ，明确接受前瞻范围的缩短。延长整段执行时间时，应独立选择与验证对应的码本和目标。</div>
<p><strong>码本与策略必须成套：</strong>新码本索引的意义已经改变，需要重新标注并训练/微调动作策略；把新码本直接替换进原 LightNav 检查点无法复现本研究结果。新研究码本的 256 个索引都用于运动编码，没有保留 L0=6 为 STOP，终止应由显式 stop 标志或 Affordance STOP token 决定。这里交付的是通过离线重建验证的量化器和 NumPy 解码器。</p>
<p><a href="research-models/{test(preferred)['config']['model']}.npz">下载优选 H5 码本</a> · <a href="research-code/decoder.py">NumPy 解码器</a> · <a href="research-deployment-contract.json">时间、坐标与终止约定</a> · <a href="research-bundle.zip">全部代码与 16 份码本</a> · <a href="research-protocol.json">研究协议</a> · <a href="research-selection_lock.json">测试前冻结记录</a> · <a href="research-exact_search_probe.json">穷举核查</a></p>
<details><summary>复现顺序与数据边界</summary><p>在具有同源发版数据的环境中运行：<code>prepare.py</code> 验证并创建几何缓存；<code>experiment.py define</code> 固定变体；<code>run_queue.py train</code> 训练；<code>run_queue.py validation</code> 评估验证集；<code>freeze.py</code> 固定选择及哈希；<code>run_queue.py test</code> 运行测试；<code>execution.py --all</code> 比较执行方式；<code>exact_probe.py</code> 核查近似搜索；<code>collect.py</code> 与 <code>make_report.py</code> 汇总。训练样本最多每场景 2,048 个完整 5 秒锚点；60 个历史展示场景全部排除。私有视频路径和 episode ID 未发布，公开拆分哈希用于核对研究来源。</p></details>
<p class="muted">每场景只取一个确定性 batch，因此是场景覆盖样本。区间按场景重采样 2,000 次并保留轨迹数权重，不能将同场景相邻帧当成独立样本；完整变体表属于探索性对照，没有对全部同时比较作多重检验校正。当前成果未覆盖视觉策略学习难度、闭环成功率、碰撞或机器人控制。</p>
<p>方法来源：<a href="https://arxiv.org/html/2608.30935#S3.SS4">LightNav-0 的残差轨迹量化目标</a> · <a href="https://github.com/lightorigins/LightNav-0/blob/main/docs/DEPLOYMENT.md">官方首点执行与时间说明</a> · <a href="https://arxiv.org/abs/2203.01941">Residual Quantization 原论文</a> · <a href="https://github.com/kakaobrain/rq-vae-transformer">RQ-VAE 作者实现</a>。本研究的新码本、特征和损失是自己的消融方案。</p></section></main>
<script>
const filters=['family','tokens','search'].map(x=>document.getElementById(x));
function applyFilters(){{let n=0;for(const row of document.querySelectorAll('#all tbody tr')){{const show=(!filters[0].value||row.dataset.family===filters[0].value)&&(!filters[1].value||row.dataset.tokens===filters[1].value)&&row.dataset.name.toLowerCase().includes(filters[2].value.toLowerCase());row.hidden=!show;if(show)n++;}}document.getElementById('visible').textContent=`显示 ${{n}} / 60 个变体`;}}
for(const input of filters)input.addEventListener('input',applyFilters);applyFilters();
</script></body></html>'''
    (site / 'research.html').write_text(document, encoding='utf-8')
    with zipfile.ZipFile(site / 'research-bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for directory in (code_dir, model_dir):
            for file in directory.iterdir(): bundle.write(file, file.relative_to(site))
        for name in ('research-deployment-contract.json', 'research-protocol.json', 'research-selection_lock.json', 'research-experiments.json'):
            bundle.write(site / name, name)
    print(site / 'research.html', flush=True)


if __name__ == '__main__':
    main()
