"""Publish aggregate RVQ research, scientific figures, and anonymized overlays."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from experiment import integrate
from prepare import hashed

BLUE, GREEN, RED = '#1676be', '#159b78', '#df5b4c'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
FONT_PATH = '/usr/share/fonts/dejavu/DejaVuSans.ttf'


def figure(path, names, values, labels, ylabel, title, colors=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(np.arange(len(names)), values, color=colors or BLUE, width=.65)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(len(names)), labels, rotation=12)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.23)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def responsive_pair(fig, path):
    """Keep chart text readable on a narrow phone screen."""
    fig.tight_layout(); fig.savefig(path, dpi=180)
    fig.set_size_inches(6.6, 8.6)
    grid = fig.add_gridspec(2, 1)
    for ax, slot in zip(fig.axes, grid):
        ax.set_subplotspec(slot)
    for label in fig.findobj(matplotlib.text.Text):
        label.set_fontsize(max(12, label.get_fontsize() * 1.25))
        if label in [ax.title for ax in fig.axes] or label is fig._suptitle:
            label.set_text('\n'.join(textwrap.wrap(label.get_text(), 25)))
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path.with_name(path.stem + '-mobile.png'), dpi=180)
    plt.close(fig)


def overlay(path, gt, k, r, rvq_label='新 5 点 RVQ', rvq_color=GREEN):
    """Share a rigid view transform; it changes neither errors nor distances."""
    angle = np.pi / 2 - gt[0, 2]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = gt[0, :2].copy()
    gt, k, r = [a.copy() for a in (gt, k, r)]
    for a in (gt, k, r):
        a[:, :2] = (a[:, :2] - origin) @ rotation.T
        a[:, 2] += angle
    w, h, pad = 1000, 720, 75
    all_xy = np.vstack([gt, k, r])[:, :2]
    middle = (all_xy.max(0) + all_xy.min(0)) / 2
    extent = np.maximum(np.ptp(all_xy, axis=0) * 1.3, 3.)
    span = float(extent.max())
    top, bottom = 108, h - 100
    scale = min((w - 2 * pad) / extent[0], (bottom - top) / extent[1])
    font = ImageFont.truetype(FONT_PATH, 23)
    small = ImageFont.truetype(FONT_PATH, 19)
    def point(x):
        return int(w / 2 + (x[0] - middle[0]) * scale), int((top + bottom) / 2 - (x[1] - middle[1]) * scale)
    images = []
    kerror, rerror = [np.linalg.norm(a[-1, :2] - gt[-1, :2]) for a in (k, r)]
    grid_m = min((.25, .5, 1, 2, 5, 10), key=lambda m: abs(m * scale - 65))
    for end in np.unique(np.linspace(0, len(gt) - 1, min(75, len(gt)), dtype=int)):
        image = Image.new('RGB', (w, h), '#f7f9fc'); d = ImageDraw.Draw(image)
        xmin, xmax = middle[0] + (np.array([pad, w - pad]) - w / 2) / scale
        ymin, ymax = middle[1] + (np.array([top, bottom]) - (top + bottom) / 2) / scale
        for x in np.arange(np.ceil(xmin / grid_m) * grid_m, xmax, grid_m):
            px = point([x, middle[1]])[0]
            d.line((px, top, px, bottom), fill='#e3e9f0')
        for y in np.arange(np.ceil(ymin / grid_m) * grid_m, ymax, grid_m):
            py = point([middle[0], y])[1]
            d.line((pad, py, w - pad, py), fill='#e3e9f0')
        d.text((pad, 16), f'俯视轨迹 · {end * .5:.1f} 秒 · 路点 {end + 1}/{len(gt)}', fill='#172033', font=font)
        d.text((pad, 52), f'最终终点偏差：K256 {kerror * 100:.1f} cm   RVQ {rerror * 100:.1f} cm', fill='#49586b', font=small)
        scale_m = min((.1, .2, .5, 1, 2, 5, 10, 20, 50), key=lambda m: abs(m * scale - 90))
        sx, sy = w - pad, 80
        d.line((sx - scale_m * scale, sy, sx, sy), fill='#49586b', width=2)
        d.line((sx, sy - 4, sx, sy + 4), fill='#49586b', width=2)
        d.line((sx - scale_m * scale, sy - 4, sx - scale_m * scale, sy + 4), fill='#49586b', width=2)
        d.text((sx - scale_m * scale, sy - 19), f'{scale_m:g} m', fill='#49586b', font=small)
        d.line([point(x) for x in gt], fill='#bac4d1', width=3)
        gx, gy = point(gt[-1]); d.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline='#242b3b', width=2)
        d.text((gx + 10, gy - 24), 'GT 终点', fill='#242b3b', font=small)
        for j, (arr, color, label) in enumerate([(gt, '#242b3b', 'GT 真实轨迹'), (k, RED, 'K256'), (r, rvq_color, rvq_label)]):
            points = [point(x) for x in arr[:end + 1]]
            if len(points) > 1: d.line(points, fill=color, width=5 if j == 0 else 3, joint='curve')
            for x, y in points:
                d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            x, y = points[-1]; d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline='white')
            angle = arr[end, 2]
            arrow = point(arr[end, :2] + np.array([np.cos(angle), np.sin(angle)]) * max(.35, span * .035))
            d.line((x, y, *arrow), fill=color, width=4)
            for turn in (-.5, .5):
                back = (int(arrow[0] - 10 * np.cos(angle + turn)), int(arrow[1] + 10 * np.sin(angle + turn)))
                d.line((*arrow, *back), fill=color, width=3)
            d.rectangle((pad + 295 * j, h - 68, pad + 295 * j + 17, h - 51), fill=color)
            d.text((pad + 295 * j + 25, h - 76), label, font=font, fill='#172033')
        ox, oy = point(gt[0]); d.rectangle((ox - 5, oy - 5, ox + 5, oy + 5), outline='#172033', width=2)
        d.text((ox + 10, oy + 6), '共同起点', fill='#49586b', font=small)
        d.text((pad, h - 36), f'箭头＝朝向 · 每格 {grid_m:g} m · 起点朝向统一朝上 · 三条轨迹使用相同尺度', fill='#58677b', font=small)
        if end == len(gt) - 1:
            for a, color in ((k, RED), (r, rvq_color)):
                p0, p1 = np.array(point(gt[-1])), np.array(point(a[-1]))
                for t in np.arange(0, 1, .16):
                    d.line((*np.round(p0 + (p1 - p0) * t).astype(int), *np.round(p0 + (p1 - p0) * min(1, t + .08)).astype(int)), fill=color, width=2)
        images.append(image)
    images[-1].save(path.with_suffix('.png'))
    durations = [130] * len(images)
    durations[0], durations[-1] = 450, 2000
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)


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
:root{--ink:#172033;--muted:#58677b;--line:#dfe7ef;--blue:#1676be;--green:#159b78;--red:#df5b4c}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.7 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
header{background:#152d42;color:white;padding:48px 22px}.wrap,main{max-width:1150px;margin:auto}main{padding:0 22px 60px}
h1{font-size:clamp(29px,4vw,46px);line-height:1.2;margin:12px 0}h2{font-size:25px;margin:0 0 12px}h3{font-size:19px;margin:12px 0 6px}p{margin:8px 0 14px}
header p{color:#d8e8f3;max-width:950px}a{color:#075eaa}header a{color:#b8e8ff}code{background:#eef3f8;padding:1px 4px;border-radius:4px;overflow-wrap:anywhere}
section,.card{background:white;border:1px solid var(--line);border-radius:14px;padding:25px;margin:19px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{margin:18px 0 0;padding:18px}.card strong{font-size:29px;display:block}
.callout{border-left:5px solid var(--green);background:#edf9f4;padding:15px 18px;border-radius:7px;margin:16px 0}.note{border-left:4px solid #dca336;background:#fff8e9;padding:14px 18px;margin:16px 0;border-radius:7px}.muted,figcaption,small{color:var(--muted)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}figure img{width:100%;height:auto;display:block}figcaption{padding:9px 12px;font-size:13px}
.scroll{overflow:auto}table{border-collapse:collapse;min-width:780px;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}thead{background:#edf3f8}td.note-col{text-align:left;min-width:150px}tbody tr:hover{background:#f5f9fc}.selected{background:#eff9f5}
.route{display:grid;grid-template-columns:.75fr 1.5fr;gap:20px;align-items:center;margin:14px 0;padding:16px;border:1px solid var(--line);border-radius:10px}.route img{width:100%;border-radius:8px}.tag{display:inline-block;background:#e9f6f1;color:#14634e;padding:3px 10px;border-radius:20px;font-size:13px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}select,input,button{font:inherit;padding:7px 12px;border:1px solid #b8c7d7;border-radius:7px}button{cursor:pointer;background:#eaf6f1;color:#14634e;font-weight:650}button:hover{background:#d7eee4}details{margin:15px 0}summary{cursor:pointer;font-weight:650;padding:12px 0}ul{padding-left:24px}li{margin:7px 0}.eyebrow{letter-spacing:.1em;color:#a9e8ce;font-size:13px}
html{scroll-behavior:smooth;scroll-padding-top:70px}.reading-nav{position:sticky;top:0;background:#fffef9f5;border-bottom:1px solid var(--line);z-index:5;backdrop-filter:blur(10px)}.reading-nav .wrap{display:flex;gap:18px;overflow:auto;padding:12px 22px;white-space:nowrap}.reading-nav a{text-decoration:none;font-size:14px}.kicker{color:#147a61;font-size:14px;font-weight:700}.lead{font-size:19px;line-height:1.75}.takeaway{font-weight:650;background:#edf9f4;padding:13px 16px;border-radius:8px;margin:12px 0}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.flow div{padding:15px;background:#eef4f9;border-radius:9px;font-size:14px}.flow strong{display:block;margin-bottom:5px}.metric-list{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric-list div{background:#f5f8fb;padding:16px;border-radius:9px}.metric-list strong{display:block}.metric-list p{font-size:14px;margin-bottom:0}.timeline{display:flex;gap:8px;align-items:stretch;overflow:auto;padding:8px 0;margin-bottom:12px}.timeline span{background:#edf3f8;min-width:90px;text-align:center;padding:10px;border-radius:8px;line-height:1.6;font-size:14px}.timeline span.execute{background:#d9f2e7;border:2px solid #159b78}.legend{display:flex;gap:20px;flex-wrap:wrap;font-weight:650}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:7px}.distribution{display:flex;height:30px;border-radius:8px;overflow:hidden;margin:14px 0}.distribution span{display:block}.category-head{padding:14px 0 4px;border-top:1px solid var(--line);margin-top:24px}.category-head p{font-size:14px}.case-reading{background:#f1f5fa;padding:10px 12px;font-size:14px;border-radius:7px}.player-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:9px}.player-controls a{font-size:13px}.appendix>details{border-top:1px solid var(--line)}.appendix>details>summary{font-size:18px}.appendix figure{margin:14px 0}.compact{min-width:620px}.text-left td,.text-left th{text-align:left}.recommendation{border:2px solid #b8ddcd}.reading-time{font-size:13px;font-weight:500;color:#58677b}.status-line{font-size:13px;color:#58677b}.route img{height:auto}.route small{display:block}.hero-note{font-size:14px}.spotlight{background:#eff9f5;border-color:#b8ddcd}.route .tag.k256_better{background:#fff0ec;color:#9f3f31}.route .tag.close{background:#edf2f8;color:#49586b}
@media(max-width:760px){.grid,.cards,.route{grid-template-columns:1fr}section{padding:18px}.card{margin-top:8px}table{font-size:13px}}
@media(max-width:760px){header{padding-top:28px;padding-bottom:28px}.spotlight{order:-1}.flow,.metric-list{grid-template-columns:1fr 1fr}.reading-nav .wrap{gap:15px;padding:10px 18px}.lead{font-size:17px}.route{padding:12px}.metric-list div:last-child{grid-column:1/-1}.wrap,main{min-width:0}main{padding-left:14px;padding-right:14px}h2{font-size:23px}.timeline span{min-width:83px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--site', type=Path, required=True)
    p.add_argument('--font', type=Path, help='Optional Chinese font for figures and GIF labels')
    args = p.parse_args()
    global FONT_PATH
    if args.font:
        FONT_PATH = str(args.font)
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams['axes.unicode_minus'] = False
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
    main_labels = ['原发版 RVQ（10 点）', 'K256', '新 RVQ（10 点差分）', '预选 RVQ（5 点）', '新 RVQ（3 点）', '新 RVQ（1 点）']
    figure(assets / 'research-overview.png', main_names, [metric(n) for n in main_names], main_labels,
           '整轨终点偏差均值（米，越低越好）', '同一批 3,760 条测试轨迹：从共同起点开环重建', [BLUE, RED, '#6d89bb', GREEN, '#52bda2', '#a4d9c4'])
    zoom = main_names[1:]
    figure(assets / 'research-precision.png', zoom, [metric(n) for n in zoom], main_labels[1:],
           '整轨终点偏差均值（米，越低越好）', '放大低误差区：路点数与前瞻范围不同', [RED, '#6d89bb', GREEN, '#52bda2', '#a4d9c4'])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (name, color, label) in enumerate([('official_release', BLUE, '原发版 RVQ'), ('k256', RED, 'K256'), (preferred, GREEN, '预选 RVQ（5 点）')]):
        values = [metric(name, key) for key in ('segment5_endpoint_m', 'segment10_endpoint_m', 'route_endpoint_m')]
        bars = ax.bar(np.arange(3) + (j - 1) * .26, values, .25, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(range(3), ['5 秒一段', '10 秒一段', '整条轨迹'])
    ax.set_ylabel('终点偏差均值（米，越低越好）'); ax.set_title('每段起点与 GT 对齐，段内累计执行；整轨只对齐一次', fontweight='bold')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    fig.tight_layout(); fig.savefig(assets / 'research-open-loop.png', dpi=170); plt.close(fig)
    horizons = [10, 5, 3, 1]
    horizon_names = [f'new_h{h}_diff_y1_beam' for h in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, unit in [(axes[0], 'route_endpoint_m', '整轨终点偏差均值（米）'), (axes[1], 'step_yaw_deg', '每步转角误差均值（度）')]:
        values = [metric(n, key) for n in horizon_names]
        bars = ax.bar([f'{h} 个路点' for h in horizons], values, color=GREEN)
        ax.bar_label(bars, fmt='%.3f', padding=3); ax.set_ylabel(unit); ax.set_ylim(0, max(values) * 1.24)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('只改路点数并重训：同为 3 个动作 token、同一表示和损失', fontweight='bold')
    responsive_pair(fig, assets / 'research-horizon.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, unit in [(axes[0], 'route_endpoint_m', '整轨终点偏差均值（米）'), (axes[1], 'step_yaw_deg', '每步转角误差均值（度）')]:
        x = np.arange(3)
        for offset, encoder, color, label in [(-.18, 'l2', '#9fb3ca', '特征误差 + 贪心选码'), (.18, 'beam', GREEN, '位置和方向损失 + 搜索选码')]:
            values = [metric(f'new_h5_diff_y{weight}_{encoder}', key) for weight in ('0.3', '1', '3')]
            bars = ax.bar(x + offset, values, .36, color=color, label=label); ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        ax.set_xticks(x, ['0.3', '1.0', '3.0']); ax.set_xlabel('训练中角度尺度（米/弧度）'); ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.suptitle('5 点 RVQ：同时检查训练尺度与编码时的选码目标', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-yaw.png', dpi=170); plt.close(fig)
    loss_names = ['official_release', 'official_mask', 'official_search64', 'official_yaw_all10', 'official_exec30', 'official_prefix5']
    figure(assets / 'research-objective.png', loss_names, [metric(n) for n in loss_names],
        ['原发版', '忽略尾部填充', '扩大搜索并细化', '约束所有点方向', '约束执行点方向', '只优化前 5 点'],
        '整轨终点偏差均值（米）', '保持原码本，改变选码方法（仍使用原 GT 路点时间戳）', [BLUE] * 4 + [GREEN] * 2)
    rep_names = ['new_h5_diff_y1_beam', 'new_h5_abs_y1_beam', 'new_h5_sincos_y1_beam', 'factorized_xy2_yaw1', 'factorized_xy1_yaw2']
    figure(assets / 'research-representation.png', rep_names, [metric(n) for n in rep_names],
        ['相邻位姿差分', '当前机体系绝对位姿', '用 sin/cos 表示方向', '2 个位置码 + 1 个角度码', '1 个位置码 + 2 个角度码'],
        '整轨终点偏差均值（米）', '5 点 RVQ：改变表示与位置/角度的码位分配（同为 3 token）')
    turn_keys = ['turn_lt1', 'turn_1to5', 'turn_5to15', 'turn_ge15']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, names, title in [(axes[0], ['official_release', 'k256', preferred], '每步转角误差均值'),
                             (axes[1], ['k256', preferred], '放大 K256 与预选方案')]:
        width = .75 / len(names)
        for j, name in enumerate(names):
            values = [test(name)[key]['yaw_deg']['mean'] for key in turn_keys]
            color = BLUE if name == 'official_release' else RED if name == 'k256' else GREEN
            label = '原发版 RVQ' if name == 'official_release' else 'K256' if name == 'k256' else '预选 5 点 RVQ'
            bars = ax.bar(np.arange(4) + (j - (len(names) - 1) / 2) * width, values, width, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
        ax.set_xticks(range(4), ['<1°', '1–5°', '5–15°', '≥15°']); ax.set_xlabel('GT 在 0.5 秒内的实际转角')
        ax.set_ylabel('误差（度）'); ax.set_title(title); ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout(); fig.savefig(assets / 'research-turn-strata.png', dpi=170); plt.close(fig)
    ex_names = ['official_release', preferred, 'new_h5_diff_y3_l2', 'new_h5_action_only']
    ex_labels = ['原发版 RVQ', '预选 RVQ（5 点）', '5 点 RVQ · 角度尺度 3', '5 点码本 · 只优化首步']
    execution = s['execution']['test']
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for ax, key, title in [(axes[0], 'endpoint_m', '终点偏差均值（米，越低越好）'), (axes[1], 'request_fraction_vs_k256', '请求数 / K256 请求数')]:
        for j, (mode, color, label) in enumerate([('step', BLUE, '每 0.5 秒重规划'), ('up_to_1s', '#8ab9c3', '一次最多执行 1 秒'), ('full_chunk', GREEN, '一次执行整段有效路点')]):
            values = [execution[n][mode][key]['mean'] if key == 'endpoint_m' else execution[n][mode][key] for n in ex_names]
            bars = ax.bar(np.arange(len(ex_names)) + (j - 1) * .26, values, .25, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=7)
        ax.set_xticks(range(len(ex_names)), ex_labels, rotation=15); ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('执行频率改变后，同样的编码结果需要重新评估', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-execution.png', dpi=170); plt.close(fig)
    prefix = s['prefix_analysis']['test']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, ylabel in [(axes[0], 'local_ade_m', '5 点局部 ADE 均值（米）'), (axes[1], 'route_endpoint_m', '整轨终点偏差均值（米）')]:
        values = [prefix[str(level)][key]['mean'] for level in (1, 2, 3)]
        bars = ax.bar(['L0', 'L0 + L1', 'L0 + L1 + L2'], values, color=['#b7d9ce', '#6ab69c', GREEN])
        ax.bar_label(bars, fmt='%.3f', padding=3); ax.set_ylabel(ylabel); ax.set_ylim(0, max(values) * 1.23)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('预选 5 点 RVQ：逐层叠加残差码的作用（同一组 token）', fontweight='bold')
    fig.tight_layout(); fig.savefig(assets / 'research-prefix.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, names, factor, unit, title in [
        (axes[0], ['official_release', 'k256', preferred], 1, '终点偏差（米）', '原发版 → 改进后：相同测试轨迹'),
        (axes[1], ['k256', preferred], 100, '终点偏差（厘米）', '放大右侧两种低误差方案')]:
        labels = ['原发版 RVQ\n10 点 · 3 token' if n == 'official_release' else 'K256\n1 点 · 1 token' if n == 'k256' else '预选新 RVQ\n5 点 · 3 token' for n in names]
        vals = [metric(n) * factor for n in names]
        bars = ax.bar(labels, vals, color=[BLUE if n == 'official_release' else RED if n == 'k256' else GREEN for n in names], width=.58)
        ax.bar_label(bars, labels=[f'{v:.3f} m' if factor == 1 else f'{v:.1f} cm' for v in vals], padding=5, fontsize=12)
        ax.set_title(title, fontweight='bold'); ax.set_ylabel(unit + '，越低越好')
        ax.set_ylim(0, max(vals) * 1.25); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    responsive_pair(fig, assets / 'research-key-results.png')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, key, title in [(axes[0], None, '全部 162,502 个动作'), (axes[1], 'in_place_turn', '其中 21,016 个原地转向动作')]:
        names = ['official_release', 'k256', preferred]
        values = [metric(n, 'step_yaw_deg') if key is None else test(n)[key]['yaw_deg']['mean'] for n in names]
        bars = ax.bar(['原发版 RVQ', 'K256', '预选 5 点 RVQ'], values, color=[BLUE, RED, GREEN], width=.58)
        ax.bar_label(bars, labels=[f'{v:.3f}°' for v in values], padding=4, fontsize=11)
        ax.set_title(title, fontweight='bold'); ax.set_ylabel('每步转角误差均值（度），越低越好')
        ax.set_ylim(0, max(values) * 1.25); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    responsive_pair(fig, assets / 'research-heading.png')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, key, title in [(axes[0], 'route_endpoint_m', '只看整轨终点：仅首步目标看起来更准'), (axes[1], 'local_ade_m', '再看未来 5 点：仅首步目标损失了前瞻精度')]:
        values = [metric(n, key) * 100 for n in (preferred, 'new_h5_action_only')]
        bars = ax.bar(['预选 5 点 RVQ', '只优化首步的 5 点码本'], values, color=[GREEN, '#ae7b3c'], width=.58)
        ax.bar_label(bars, labels=[f'{v:.1f} cm' for v in values], padding=5, fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=11); ax.set_ylabel('偏差均值（厘米），越低越好')
        ax.set_ylim(0, max(values) * 1.25); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    responsive_pair(fig, assets / 'research-tradeoff.png')
    cards, categories = gallery(args.cache, root, assets, preferred)
    (site / 'research-gallery.json').write_text(json.dumps({'cases': cards, 'endpoint_gap_threshold_m': .05, 'counts': categories}, indent=2) + '\n')
    category_names = {'rvq_better': '5 点 RVQ 占优', 'close': '终点误差接近', 'k256_better': 'K256 占优'}
    def case_card(i, g):
        gap = g['rvq_endpoint_m'] - g['k256_endpoint_m']
        along = g['rvq_ade_m'] - g['k256_ade_m']
        endpoint_reading = ('两者终点偏差相差不到 5 cm' if abs(gap) <= .05 else
                            f"{'RVQ' if gap < 0 else 'K256'} 的终点偏差少 {abs(gap) * 100:.1f} cm")
        reading = f"{endpoint_reading}；{'RVQ' if along < 0 else 'K256'} 的途中平均偏差也{'更小' if (along < 0) == (gap < 0) else '较小'}。"
        if abs(gap) <= .05:
            reading = endpoint_reading + f"；途中平均偏差以 {'RVQ' if along < 0 else 'K256'} 更小。"
        still = g['asset'].replace('.gif', '.png')
        return f'''<article class="route" id="case-{i}"><div><span class="tag {g['category']}">{category_names[g['category']]}</span>
<h3>案例 {i} · GT 走了 {g['gt_length_m']:.1f} 米</h3>
<p><strong>终点偏差</strong><br>K256 {g['k256_endpoint_m'] * 100:.1f} cm ／ RVQ {g['rvq_endpoint_m'] * 100:.1f} cm<br><strong>途中平均偏差（整轨 ADE）</strong><br>K256 {g['k256_ade_m'] * 100:.1f} cm ／ RVQ {g['rvq_ade_m'] * 100:.1f} cm</p>
<p class="case-reading">怎么看：{reading}</p><p><strong>绘制路点数</strong><br>K256：{g['k256_waypoints']} 个<br>RVQ：{g['rvq_waypoints']} 个<br>GT：{g['gt_waypoints']} 个</p>
<small>均含起点；每 0.5 秒增加一个执行点。RVQ 每次内部编码的 5 点是重叠前瞻窗。</small></div>
<div><img id="route-image-{i}" loading="lazy" width="1000" height="720" src="assets/{still}" alt="案例{i}最终俯视图：黑色GT、红色K256、绿色新5点RVQ，共同起点、米制比例尺和朝向箭头">
<div class="player-controls"><button type="button" data-player="route-image-{i}" data-gif="assets/{g['asset']}" data-still="assets/{still}" aria-pressed="false">播放轨迹动图</button><a href="assets/{still}" target="_blank" rel="noopener">放大最终图</a><a href="assets/{g['asset']}">原始 GIF</a></div></div></article>'''
    gallery_parts = []
    category_reading = {'rvq_better': '绿色方案的终点偏差比红色至少少 5 cm。', 'close': '两者终点偏差相差不超过 5 cm，继续比较途中偏差。', 'k256_better': '红色方案的终点偏差比绿色至少少 5 cm，改进后的 RVQ 仍有失败案例。'}
    for kind, count in categories.items():
        cases = [(i, g) for i, g in enumerate(cards, 1) if g['category'] == kind]
        gallery_parts.append(f'<div class="category-head"><h3>{category_names[kind]} · {count:,} 条 / {count / test(preferred)["routes"]:.1%}</h3><p>{category_reading[kind]} 以下先看一条，再展开另一种路程的例子。</p></div>')
        gallery_parts.append(case_card(*cases[0]))
        gallery_parts.append(f'<details><summary>再看一条 {cases[1][1]["gt_length_m"]:.1f} 米的例子</summary>{case_card(*cases[1])}</details>')
    gallery_html = ''.join(gallery_parts)
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
    appendix = f'''<details><summary>完整指标主表：预算、前瞻与更多候选</summary><div class="callout">按验证集事先选定的 5 点、3 token 方案，在新测试集的整轨终点误差均值为 <strong>{metric(preferred):.4f} m</strong>，相比原发版 RVQ 的 {metric('official_release'):.4f} m 降低 {reduction_old:.1%}，相比同批 K256 的 {metric('k256'):.4f} m 降低 {reduction_k:.1%}。RVQ−K256 的配对均值差 95% 场景 bootstrap 区间为 [{ci[0]:.4f}, {ci[1]:.4f}] m。</div>
<p>该方案使用固定 0.5 秒时间网格、5 个未来局部 SE(2) 位姿、三层各 256 项的 RVQ。码本在训练场景上拟合；编码时同时优化整个有效窗口的位置、各点方向与本次执行点方向。窗口 ADE 为 {preferred_test['local_ade_m']['mean']:.4f} m，P95 为 {preferred_test['local_ade_m']['p95']:.4f} m，保留未来几何信息。预算是每次 3 个动作 token，K256 为 1 个；两者提供的信息量不同。优选来自验证集，测试集打开前已冻结所有变体与码本哈希。</p>
<div class="grid"><figure><img loading="lazy" src="assets/research-overview.png" alt="原RVQ、K256与不同路点数新RVQ的终点误差柱状图"><figcaption>相同的 3,760 条测试轨迹；所有终点误差均从共同 GT 起点开环积分。</figcaption></figure><figure><img loading="lazy" src="assets/research-precision.png" alt="固定时间RVQ与K256精度对比柱状图"><figcaption>放大低误差区间。1 点方案只保留下一步，不能据此宣称 5 点或 10 点未来预测同样精确。</figcaption></figure></div>
<div class="scroll"><table><thead><tr><th>用法</th><th>动作 token</th><th>内部路点</th><th>整轨终点均值 m</th><th>终点 P95 m</th><th>单步方向 °</th><th>自身窗口 ADE m</th></tr></thead><tbody>{main_rows}</tbody></table></div>
<p class="muted">不同 H 的“自身窗口 ADE”覆盖范围不同；比较 5 点与 10 点时，下方另给相同前 5 点 ADE。表中 token 数只计算动作；Pointing 的 Object / Affordance 仍另占 2 个 token。</p></details>
<details><summary>另一候选：额外约束首点位置及三个随机种子</summary><p><code>new_h5_control_weighted</code> 在 H5 差分损失中额外加入首点位置误差，仍保留整个窗口 ADE。其验证集终点均值为 {metric('new_h5_control_weighted', split='validation'):.4f} m，测试为 {metric('new_h5_control_weighted'):.4f} m，局部窗口 ADE 为 {metric('new_h5_control_weighted','local_ade_m'):.4f} m；三个种子的测试结果为 {metric('new_h5_control_weighted'):.4f}、{metric('new_h5_control_seed2'):.4f}、{metric('new_h5_control_seed3'):.4f} m。这是一种保留前瞻且开环误差小的候选用法。</p><p>它在验证集上的排序略落后于绝对坐标 H5，因此保留测试前冻结的优选标记。报告同时展示其更低的测试观察值，供后续在实际任务验证集上比较；没有根据本次测试排名重新挑选种子或更新码本。</p></details>
<details><summary>数据与实现审计：源身份、坐标、时间、填充、解码复现</summary><ul>
<li>每条轨迹核对源视频身份、帧数、GT 位姿、2 Hz 时间轴和 K256 STOP 约定；每个 batch 校验原标注 SHA 收据。只读几何，RGB 读取为 0。</li>
<li>测试保留全部非终止动作，没有按旧 RVQ 的 training_eligible 标签或量化误差筛选。所有变体使用相同的轨迹和动作。</li>
<li>原发版编码使用当前帧 GT 起点的重叠未来窗；3 个码字求和后解码，不是每个 token 各编码一段轨迹。原几何细分窗以距离≤0.25 m、转角≤30°取前 10 点，0.5 秒执行点的索引会随细分数变化。</li>
<li>新固定时间方案的第一个路点始终对应下一源帧，既不需要 GT 侧车时间戳，也不把不足 H 点的尾部填充纳入编码损失和指标。0.25 秒变体只插值原 2 Hz GT，没有增加真实观测。</li>
<li>无量化 GT 增量重建在测试集的最大终点误差为 {s['audits']['identity_max_endpoint_m']['test']:.2e} m。发版编码器在完整验证集重编码 token 一致率为 {s['audits']['validation_encoder_reproduction']['same_code_tuple_fraction']:.2%}；优选方案的独立 NumPy 解码器验证 {s['audits']['independent_numpy_decoder']['windows']:,} 个窗口，位置最大差 {s['audits']['independent_numpy_decoder']['max_xy_abs_difference_m']:.2e} m。</li>
</ul><p>编码目标由各帧 GT 未来窗口生成，回放时将解码动作应用在当前已重建位姿上，整轨仅在初始位姿对齐。整轨指标对轨迹等权；单步指标对动作等权，局部 ADE 先在有效点内取均值，再对窗口等权。5 秒与 10 秒实验分别每 10、20 个动作从 GT 起点重置，段内仍开环积分。</p>
<figure><img loading="lazy" src="assets/research-open-loop.png" alt="5秒、10秒和整轨开环终点误差柱状图"><figcaption>不同开环时长均比较同一批可用段。完整段数和分位数记录在机器可读结果中。</figcaption></figure>
<p class="muted">测试集重编码有 {s['audits']['test_encoder_near_ties']['different_code_windows']} / {preferred_test['steps']:,} 个近似并列窗口选择了不同 token，损失差仅 {s['audits']['test_encoder_near_ties']['max_objective_difference']:.2e}，导致整轨终点均值差 {s['audits']['test_encoder_near_ties']['mean_route_endpoint_delta_m']:.2e} m。原发版 baseline 始终使用保存的原 token，未以重编码替代。</p><div class="note">本研究衡量 GT 轨迹经过量化后的重建误差。没有运行视觉策略、碰撞检测、机器人动力学或反馈控制，因此这些数字用于选择编码方案，不是导航成功率。原 K256 码本曾使用全保留集拟合，本次新测试场景对它不构成严格训练隔离；新增 RVQ 码本则只接触 64 个训练场景。表中的 <code>new_h1_1token</code> 提供了同训练样本的单层 VQ 对照。</div></details>
<details><summary>角度消融细节：训练尺度、表示方法、转向分层</summary><p>角度权重从 0.3、1 到 3 m/rad 分别训练；每份码本同时比较“特征空间贪心残差分配”和“解码后几何损失 + 执行点方向”的搜索。增加角度权重会改变平移与方向的容量分配；结果并非权重越大就所有指标越好。</p>
<p>RVQ 的核心是 <code>ẑ = C₀[k₀] + C₁[k₁] + C₂[k₂]</code>：先拟合第一层码本，再对第一层留下的残差拟合第二层，再拟合第三层残差。本研究用 residual k-means、每层 256 项、k-means++ 初始化与两次重启训练；改变的是特征表示、尺度和编码分配目标，三个 token 并不各自对应一段独立轨迹。</p>
<div class="grid"><figure><img loading="lazy" src="assets/research-yaw.png" alt="RVQ角度权重和编码目标对方向及终点误差的影响"><figcaption>同一 H5 码本内对照分配目标，横向比较角度训练尺度。</figcaption></figure><figure><img loading="lazy" src="assets/research-representation.png" alt="SE2差分、绝对坐标、sin cos以及分离角度码位的对比"><figcaption>sin/cos 避免角度坐标的 ±π 跳变；本批数据中它没有明显超过绝对角度表示。分开分配码位也没有自动获得最好结果。</figcaption></figure></div>
<p>优选方案将 <code>[x₁,y₁,θ₁,…,x₅,y₅,θ₅]</code> 联合 RVQ 编码，解码输出为当前机体坐标下的绝对局部位姿。编码分配损失为 <code>ADE_valid + 0.3 × mean|wrap(角度误差)| + 1.0 × |wrap(首点角度误差)|</code>；角度以 rad 计。该目标保留整个窗口位置精度，同时直接约束每 0.5 秒执行的方向。</p>
<figure><img loading="lazy" src="assets/research-turn-strata.png" alt="按真实转向幅度分层的K256、原RVQ和优选RVQ角度误差柱状图"><figcaption>在相同 GT 转向组内比较，右图放大 K256 与优选方案，避免大误差柱掩盖细节。</figcaption></figure>
<p>对 {preferred_test['in_place_turn']['yaw_deg']['n']:,} 个原地转向动作（水平移动小于 2 cm、转角至少 5°），原 RVQ 的平均方向误差为 {original['in_place_turn']['yaw_deg']['mean']:.3f}°，K256 为 {k['in_place_turn']['yaw_deg']['mean']:.3f}°，优选 H5 为 {preferred_test['in_place_turn']['yaw_deg']['mean']:.3f}°，其 P95 为 {preferred_test['in_place_turn']['yaw_deg']['p95']:.3f}°。这些动作几乎没有 XY 位移，直接用轨迹切线替代显式角度会丢失转向信息。</p>
<p>另有“4 个联合残差 token”和“原 RVQ + 8 bit 角度修正 token”实验，全部显式计为 4 个动作 token。后者只修正被执行点的方向，没有凭空增加整个未来窗口的精度；完整表中保留它的误差、角度饱和数和额外成本。</p></details>
<details><summary>选码搜索与穷举：保留原码本能改善多少</summary><figure><img loading="lazy" src="assets/research-objective.png" alt="保持原码本时改变填充、搜索宽度与方向目标的开环结果"><figcaption>这组使用原几何采样和 GT 侧车时间戳，只用于隔离编码分配目标。它们不能直接当作固定时间部署方案。</figcaption></figure>
<p>原目标主要看窗口位置与末点方向。宽度 64 搜索和逐层细化更充分地优化该目标，却没有解决逐帧开环漂移；加入执行点方向、或优化前 5 点，改善更明显。对固定时间网格，直接复用原码本仍可能因运动范围和采样分布不同而出现较大位置误差，见 <code>official_fixed05*</code>。因此需要同时检查目标、采样与码本分布。</p>
<p>额外对 64 个验证窗口做完整三层组合穷举，以区分 RVQ 容量和近似搜索误差。搜索访问每个允许的组合；下表的“最优”只指该窗口的给定损失，容差为 10⁻⁵。</p>
<div class="scroll"><table><thead><tr><th>编码器</th><th>每窗口穷举组合数</th><th>近似搜索达到最优比例</th><th>平均损失差</th></tr></thead><tbody>{exact_rows}</tbody></table></div><p class="muted">损失含位置与加权角度，不能把损失差直接解释为纯位置误差。该小样本核查没有证明整个测试集或整条轨迹的全局最优。</p></details>
<details><summary>执行频率完整对照：误差、请求数与实际时长</summary><figure><img loading="lazy" src="assets/research-execution.png" alt="同一码本逐帧、每一秒和整段执行的误差与请求数柱状图"><figcaption>三种执行方式使用相同发版/重编码 token，按共同 2 Hz 网格回放。整段的实际时长由有效路点决定。</figcaption></figure>
<div class="scroll"><table><thead><tr><th>编码方案</th><th>逐帧终点 m</th><th>最多1秒终点 m</th><th>整段终点 m</th><th>整段平均执行 s</th><th>整段请求/K256</th></tr></thead><tbody>{execution_rows}</tbody></table></div>
<p>为首点精度优化的编码，后续方向未必同样准确；降低请求数时必须重新检查整轨漂移。只优化下一步的 H5，其逐帧结果为 {metric('new_h5_action_only'):.4f} m，但自身窗口 ADE 为 {metric('new_h5_action_only','local_ade_m'):.4f} m，直接整段执行的误差为 {execution['new_h5_action_only']['full_chunk']['endpoint_m']['mean']:.3f} m。它适合研究单步控制的精度上限，不能以“5 点”名称替代未来轨迹质量检查。完整执行结果同时记录速度 jerk、方向角加速度和请求数。</p></details>
<details><summary>残差层与随机种子</summary><figure><img loading="lazy" src="assets/research-prefix.png" alt="优选RVQ只解码第一层、两层和三层时的误差柱状图"><figcaption>截取同一组三 token 分配的前缀，展示残差层作用；没有为短前缀重新选择最优 token。</figcaption></figure>
<p>对 H5 差分基线使用相同训练锚点、三个不同 k-means 随机种子，测试终点误差均值分别为 {', '.join(f'{x:.4f}' for x in seed_endpoints)} m，均值 {np.mean(seed_endpoints):.4f} m、样本标准差 {np.std(seed_endpoints, ddof=1):.4f} m。三者实际都使用前三层；首个文件另存的第 4 层是后续追加训练，不会回写前三层，仅供 4 token 变体使用。控制加权版本也保留三个种子结果。绝对坐标优选码本是预先冻结的单个模型；场景 bootstrap 区间描述这份模型的数据抽样不确定性，不代表全部训练随机性。</p></details>
<details id="all-experiments"><summary>全部 60 个编码变体（可筛选）</summary><p>每个变体都完成了相同的 1,992 条验证轨迹和 3,760 条测试轨迹。表格展示逐帧执行结果，保留未改善、增加 token、依赖 GT 时间侧车以及牺牲前瞻精度的实验。测试结果没有用于重新选择或训练码本。</p>
<div class="controls"><label>系列 <select id="family"><option value="">全部</option>{option_html}</select></label><label>动作 token <select id="tokens"><option value="">全部</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label><label>搜索 <input id="search" placeholder="H5 / yaw / official"></label></div><p id="visible" class="muted"></p>
<div class="scroll"><table id="all"><thead><tr><th>变体</th><th>token</th><th>H</th><th>验证终点均值 m</th><th>测试终点均值 m</th><th>测试终点P95 m</th><th>单步方向 °</th><th>单步位置 m</th><th>自身窗口 ADE m</th><th>说明</th></tr></thead><tbody>{all_rows}</tbody></table></div>
<p class="muted">各系列含不同信息预算与时间约定。不同 H 的窗口 ADE 不等覆盖范围；新增 token 方案不作为同预算胜例。完整置信区间、转向分层、首5点同时间 ADE、方向残差相关性、码本哈希和执行结果见 <a href="research-summary.json">research-summary.json</a>。</p></details>
<details><summary>下载、使用约定与复现步骤</summary><div class="callout">保留 5 点未来信息时，优先采用本次验证集选出的固定 0.5 秒 H5、3 层 RVQ 与执行点方向损失，并逐帧执行首点。若只需要下一动作，可选 H1 的 2 或 3 层 RVQ，明确接受前瞻范围的缩短。延长整段执行时间时，应独立选择与验证对应的码本和目标。</div>
<p><strong>码本与策略必须成套：</strong>新码本索引的意义已经改变，需要重新标注并训练/微调动作策略；把新码本直接替换进原 LightNav 检查点无法复现本研究结果。新研究码本的 256 个索引都用于运动编码，没有保留 L0=6 为 STOP，终止应由显式 stop 标志或 Affordance STOP token 决定。这里交付的是通过离线重建验证的量化器和 NumPy 解码器。</p>
<p><a href="research-models/{test(preferred)['config']['model']}.npz">下载优选 H5 码本</a> · <a href="research-code/decoder.py">NumPy 解码器</a> · <a href="research-deployment-contract.json">时间、坐标与终止约定</a> · <a href="research-bundle.zip">全部代码与 16 份码本</a> · <a href="research-protocol.json">研究协议</a> · <a href="research-selection_lock.json">测试前冻结记录</a> · <a href="research-exact_search_probe.json">穷举核查</a></p>
<details><summary>复现顺序与数据边界</summary><p>在具有同源发版数据的环境中运行：<code>prepare.py</code> 验证并创建几何缓存；<code>experiment.py define</code> 固定变体；<code>run_queue.py train</code> 训练；<code>run_queue.py validation</code> 评估验证集；<code>freeze.py</code> 固定选择及哈希；<code>run_queue.py test</code> 运行测试；<code>execution.py --all</code> 比较执行方式；<code>exact_probe.py</code> 核查近似搜索；<code>collect.py</code> 与 <code>make_report.py</code> 汇总。训练样本最多每场景 2,048 个完整 5 秒锚点；60 个历史展示场景全部排除。私有视频路径和 episode ID 未发布，公开拆分哈希用于核对研究来源。</p></details>
<p class="muted">每场景只取一个确定性 batch，因此是场景覆盖样本。区间按场景重采样 2,000 次并保留轨迹数权重，不能将同场景相邻帧当成独立样本；完整变体表属于探索性对照，没有对全部同时比较作多重检验校正。当前成果未覆盖视觉策略学习难度、闭环成功率、碰撞或机器人控制。</p>
<p>方法来源：<a href="https://arxiv.org/html/2608.30935#S3.SS4">LightNav-0 的残差轨迹量化目标</a> · <a href="https://github.com/lightorigins/LightNav-0/blob/main/docs/DEPLOYMENT.md">官方首点执行与时间说明</a> · <a href="https://arxiv.org/abs/2203.01941">Residual Quantization 原论文</a> · <a href="https://github.com/kakaobrain/rq-vae-transformer">RQ-VAE 作者实现</a>。本研究的新码本、特征和损失是自己的消融方案。</p></details>'''
    quick_rows = ''.join(f'''<tr><th>{label}</th><td>{metric(n) * 100:.1f}</td><td>{test(n)['route_endpoint_m']['p95'] * 100:.1f}</td><td>{metric(n, 'step_yaw_deg'):.3f}°</td><td>{test(n)['config']['tokens']} 个 / {test(n)['config']['horizon']} 点</td></tr>''' for n, label in [('official_release', '原发版 RVQ'), ('k256', 'K256'), (preferred, '预选新 5 点 RVQ')])
    quick_horizon_rows = ''.join(f'''<tr><th>{h} 点</th><td>{metric(f'new_h{h}_diff_y1_beam') * 100:.1f} cm</td><td>{h * .5:.1f} 秒</td><td>{test(f'new_h{h}_diff_y1_beam')['window_travel_m']['mean']:.3f} m</td><td>{'—' if h < 5 else format(test(f'new_h{h}_diff_y1_beam')['shared_first5_ade_m']['mean'] * 100, '.2f') + ' cm'}</td></tr>''' for h in horizons)
    quick_execution_rows = ''.join(f'''<tr><th>{label}</th><td>{execution[preferred][mode]['endpoint_m']['mean'] * 100:.1f} cm</td><td>{execution[preferred][mode]['request_fraction_vs_k256']:.0%}</td></tr>''' for mode, label in [('step', '每 0.5 秒执行首点'), ('up_to_1s', '一次最多执行 1 秒'), ('full_chunk', '一次执行整段有效路点')])
    distribution = ''.join(f'<span style="width:{count / preferred_test["routes"] * 100:.4f}%;background:{color}" title="{category_names[name]} {count / preferred_test["routes"]:.1%}"></span>' for (name,count),color in zip(categories.items(), [GREEN, '#9aa8bc', RED]))
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五分钟读懂 RVQ 开环精度研究 · HM3D</title><meta name="description" content="先看结论，再看路点数、角度、执行方式三组实验与分类轨迹。60个RVQ变体的完整结果、码本与复现材料。"><style>{CSS}</style></head><body>
<header><div class="wrap"><div class="eyebrow">HM3D · RVQ 编码研究 · 2026-10-15</div><h1>RVQ 的开环误差，可以降下来吗？</h1><p>RVQ（残差向量量化）与 K256 是两种动作压缩方式；这里测把真实轨迹压缩再还原后的偏差。</p><p class="lead">可以。在本批离线重建测试中，保留 5 点前瞻的新 RVQ，把整轨终点偏差从 <strong>2.67 米降到 15.5 厘米</strong>。先用五分钟看懂结果，再按需展开实验细节。</p><a href="index.html">原发版对照报告</a> · <a href="research-summary.json">完整数据</a> · <a href="research-bundle.zip">代码与码本</a></div></header>
<nav class="reading-nav" aria-label="五分钟阅读顺序"><div class="wrap"><a href="#conclusion">① 结论</a><a href="#method">② 怎么测</a><a href="#horizon">③ 编几路点</a><a href="#heading">④ 怎么编码方向</a><a href="#execution">⑤ 怎么执行</a><a href="#trajectories">⑥ 看轨迹</a><a href="#evidence">完整证据</a></div></nav><main>
<section id="conclusion"><div class="kicker">先看结论 <span class="reading-time">· 约 30 秒</span></div><h2>保留 5 点前瞻，RVQ 的重建精度可以明显提高</h2>
<div class="cards"><div class="card"><small>原发版 RVQ · 整轨终点偏差</small><strong>{metric('official_release'):.2f} 米</strong><small>空间细分 10 点 · 3 个动作 token</small></div><div class="card"><small>K256 · 相同测试轨迹</small><strong>{metric('k256') * 100:.1f} 厘米</strong><small>下一步 1 点 · 1 个动作 token</small></div><div class="card spotlight"><small>预选新 RVQ · 相同测试轨迹</small><strong>{metric(preferred) * 100:.1f} 厘米</strong><small>固定时间 5 点 · 3 个动作 token</small></div></div>
<p class="takeaway">新 RVQ 相比原发版降低 {reduction_old:.1%}，相比 K256 降低 {reduction_k:.1%}。这是同一批轨迹上的编码重建结果，说明低精度并非 RVQ 无法改善的固有限制。</p>
<figure><a href="assets/research-key-results-mobile.png" target="_blank" rel="noopener" title="点击放大图表"><picture><source media="(max-width:760px)" width="1188" height="1548" srcset="assets/research-key-results-mobile.png"><img width="2070" height="810" src="assets/research-key-results.png" alt="整轨终点偏差：原RVQ 2.67米、K256 31.6厘米、预选新5点RVQ 15.5厘米；右侧放大K256与新RVQ"></picture></a><figcaption>柱越低越好。两图均从 0 起画；左图单位是米，右图放大为厘米。所有值为同一批测试轨迹的均值。</figcaption></figure>
<details><summary>看稳健性与成本：95%分位、转角误差、token 数</summary><div class="scroll"><table class="compact"><thead><tr><th>方案</th><th>终点均值 cm</th><th>终点95%分位 cm</th><th>每步转角误差</th><th>动作 token / 前瞻点数</th></tr></thead><tbody>{quick_rows}</tbody></table></div><p>95%分位表示 95% 的轨迹终点偏差不超过该值。新 RVQ−K256 的配对均值差 95% 场景重采样区间为 [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}] cm。token 数只计运动，Pointing 的 Object / Affordance 另占 2 个 token。</p></details>
<p class="hero-note"><strong>结论范围：</strong>这页是 GT 轨迹量化后的离线重建，尚未验证视觉策略或闭环导航。新 RVQ 使用 64 / 12 / 24 个互斥训练 / 验证 / 测试场景；K256 曾在全保留集拟合，且每步只用 1 个动作 token，两者训练数据和信息预算不同。</p>
</section>
<section id="method"><div class="kicker">实验怎么做 <span class="reading-time">· 约 45 秒</span></div><h2>把真实动作压缩，再解压，看看误差会积累多少</h2>
<div class="flow"><div><strong>1. 取相同 GT</strong>GT 就是真实轨迹；每帧间隔 0.5 秒。</div><div><strong>2. 编码 → 解码</strong>同一段真实未来运动，分别变成 K256 或 RVQ token，再还原。</div><div><strong>3. 累计执行</strong>从同一真实起点出发，把每步还原动作接到上一重建位姿。</div><div><strong>4. 比较偏差</strong>整轨只在开始对齐一次，中途不拉回 GT。</div></div>
<p>比较 <strong>{preferred_test['routes']:,} 条轨迹 / {preferred_test['scenes']} 个测试场景</strong>，平均每条 GT 走 {preferred_test['route_length_m']['mean']:.2f} 米；每种编码都使用相同的 {preferred_test['steps']:,} 个动作。60 个变体在测试前已冻结，推荐方案由验证集选出。</p>
<p class="hero-note">新 5 点方案每 0.5 秒编码一窗未来路点，只执行首点，再从下一帧生成新的重叠窗口。token 可理解为选中的码本编号。</p><div class="metric-list"><div><strong>整轨终点偏差</strong><p>一路累计后，最后离真实终点多远？用它判断长期漂移。</p></div><div><strong>局部 ADE</strong><p>当前帧内，各未来路点离 GT 的距离取平均。用它判断前瞻形状，尚未累计整轨误差。</p></div><div><strong>每步转角误差</strong><p>一个 0.5 秒动作的转角与真实转角相差几度？小误差反复出现也会造成漂移。</p></div></div>
<details><summary>RVQ 是每 10 点编码一次，还是每帧都编码？</summary><p><strong>每一帧都生成重叠的未来窗口。</strong>三个 RVQ token 是三层码字叠加，共同还原一整窗路点；每个 token 都参与所有点的重建。</p><p>原发版：把未来源轨迹按每段位移 ≤0.25 m、转角 ≤30°细分，取前 10 点；路点间隔不固定。评测通过已保存的 GT 时间戳找到 0.5 秒执行点。</p><p>新 5 点方案：路点固定在未来 0.5、1、1.5、2、2.5 秒；当前帧执行第 1 点。0.5 秒后，下一帧再编码一窗，窗口向前滑动一个源帧。</p><div class="timeline"><span>当前帧<br>起点</span><span class="execute">+0.5 秒<br>第 1 点 · 执行</span><span>+1.0 秒<br>第 2 点</span><span>+1.5 秒<br>第 3 点</span><span>+2.0 秒<br>第 4 点</span><span>+2.5 秒<br>第 5 点</span></div><p>这个实验里，每帧的编码目标都来自 GT，尚未让视觉模型预测这些 token。轨迹尾部不足一窗时只统计有效点。</p></details>
</section>
<section id="horizon"><div class="kicker">关键实验 1 <span class="reading-time">· 约 45 秒</span></div><h2>同样 3 个 token、固定时间取点：本组 5 点比 10 点更准</h2><p><strong>改了什么：</strong>只改变编码路点数，并为每种长度重训码本。统一用 3 个动作 token、固定 0.5 秒间隔、差分位姿表示、相同训练锚点与编码损失。本组 10 点使用新固定时间码本，原发版的空间细分 10 点另列在首页。</p>
<figure><a href="assets/research-horizon-mobile.png" target="_blank" rel="noopener" title="点击放大图表"><picture><source media="(max-width:760px)" width="1188" height="1548" srcset="assets/research-horizon-mobile.png"><img loading="lazy" width="1870" height="765" src="assets/research-horizon.png" alt="控制其他条件后，10、5、3、1点RVQ的整轨终点偏差和每步角度误差"></picture></a><figcaption>10 点→5 点：整轨终点偏差从 40.4 cm 降到 16.6 cm。此处是控制变量的差分方案；首页预选方案另采用局部绝对位姿，结果为 15.5 cm。</figcaption></figure>
<div class="scroll"><table class="compact"><thead><tr><th>编码路点</th><th>整轨终点偏差</th><th>最多前瞻</th><th>局部窗平均走多远</th><th>共同前5点 ADE</th></tr></thead><tbody>{quick_horizon_rows}</tbody></table></div>
<p class="takeaway">同样 3 个 token，表示更短的未来窗口，可以提高每点精度。1 点最准确，但只保留 0.5 秒前瞻；本次保留 5 点作为精度与前瞻范围的折中。</p>
<p class="hero-note"><strong>局部窗口平均走多远：</strong>原发版空间 10 点为 {original['window_travel_m']['mean']:.3f} m；新固定时间 10 点为 {test('new_h10_diff_y1_beam')['window_travel_m']['mean']:.3f} m；预选新 5 点为 {preferred_test['window_travel_m']['mean']:.3f} m。</p><details><summary>局部 10 路点 ADE 的平均距离，具体怎么算？</summary><div class="scroll"><table class="compact"><thead><tr><th>明确的采样方案</th><th>局部 ADE 均值</th><th>GT 窗口平均路程</th><th>时间约定</th></tr></thead><tbody><tr><th>原发版 · 空间细分 10 点</th><td>{metric('official_release','local_ade_m') * 100:.2f} cm</td><td>{original['window_travel_m']['mean']:.3f} m</td><td>不固定</td></tr><tr><th>新差分 · 固定时间 10 点</th><td>{metric('new_h10_diff_y1_beam','local_ade_m') * 100:.2f} cm</td><td>{test('new_h10_diff_y1_beam')['window_travel_m']['mean']:.3f} m</td><td>最多 5 秒</td></tr><tr><th>预选新 RVQ · 固定时间 5 点</th><td>{metric(preferred,'local_ade_m') * 100:.2f} cm</td><td>{preferred_test['window_travel_m']['mean']:.3f} m</td><td>最多 2.5 秒</td></tr></tbody></table></div><p>路程按“当前帧→第 1 点→第 2 点→后续有效点”的相邻水平段累加，含首段、忽略终点填充，再对 {preferred_test['steps']:,} 个窗口平均。原地转向也在分母中。它是沿轨迹走过的距离，不是起终点直线距离。</p><p>表中自身窗口的范围不同；要公平比较新 H10 与新 H5 的局部位置精度，请看上表同时间前 5 点的 ADE。</p></details>
</section>
<section id="heading"><div class="kicker">关键实验 2 <span class="reading-time">· 约 40 秒</span></div><h2>角度必须照顾本次要执行的点</h2><p><strong>改了什么：</strong>原目标强调整窗位置与最后一点的方向；新方案在编码选 token 时，同时约束各点方向，并额外惩罚本次实际执行的第 1 点的转角误差。仍保留整窗位置精度。</p>
<figure><a href="assets/research-heading-mobile.png" target="_blank" rel="noopener" title="点击放大图表"><picture><source media="(max-width:760px)" width="1188" height="1548" srcset="assets/research-heading-mobile.png"><img loading="lazy" width="2070" height="810" src="assets/research-heading.png" alt="全部动作与原地转向动作的角度误差：原RVQ、K256、预选新5点RVQ"></picture></a><figcaption>每步转角误差从原发版 {metric('official_release','step_yaw_deg'):.2f}° 降至预选方案 {metric(preferred,'step_yaw_deg'):.2f}°。原地转向定义：0.5 秒内移动小于 2 cm，且真实转角至少 5°。</figcaption></figure>
<p class="takeaway">方向精度可以显著提高。同一份新 H5 差分码本，把“特征贪心选码”改为“位置、方向与执行点联合损失 + 搜索选码”，每步转角误差由 {metric('new_h5_diff_y1_l2','step_yaw_deg'):.3f}° 降到 {metric('new_h5_diff_y1_beam','step_yaw_deg'):.3f}°。损失目标和选码方式都值得检查。</p>
<p class="hero-note">把角度训练权重无限加大并不保证更好；分开编码 XY / 角度、sin/cos、额外角度 token 都已评估，完整结果在下方证据区。仅从 XY 切线推角度会遗漏原地转向。</p>
</section>
<section id="execution"><div class="kicker">关键实验 3 <span class="reading-time">· 约 40 秒</span></div><h2>误差小，还要确认未来路点可用、执行节奏匹配</h2>
<p><strong>做了什么对照：</strong>让同一 5 点码本只优化下一步。它的整轨终点看起来很好，但后面 4 点的几何质量明显下降。</p>
<figure><a href="assets/research-tradeoff-mobile.png" target="_blank" rel="noopener" title="点击放大图表"><picture><source media="(max-width:760px)" width="1188" height="1548" srcset="assets/research-tradeoff-mobile.png"><img loading="lazy" width="2070" height="810" src="assets/research-tradeoff.png" alt="只优化首步虽把终点误差降到2厘米，但未来5点局部ADE升至24厘米"></picture></a><figcaption>两图均为厘米、越低越好。仅首步目标的终点偏差约 2 cm，但局部 5 点平均偏差约 24 cm；不能同时声称具备高精度 5 点前瞻。</figcaption></figure>
<div class="scroll"><table class="compact"><thead><tr><th>预选新 5 点 RVQ：如何执行</th><th>整轨终点偏差</th><th>请求数 / K256</th></tr></thead><tbody>{quick_execution_rows}</tbody></table></div>
<p class="takeaway">本次推荐每 0.5 秒执行首点、再规划。一次执行整窗虽然减少请求，误差会改变；只优化首步的方案若整窗执行，终点偏差达到 {execution['new_h5_action_only']['full_chunk']['endpoint_m']['mean']:.2f} 米。</p><p class="hero-note">这里的“请求数”按回放节奏计数，不是实测模型耗时。原发版→新 5 点同时改变采样、目标与码本；94.2% 的总改善不能全部归因于减少路点数。</p>
</section>
<section id="trajectories"><div class="kicker">同图看轨迹 <span class="reading-time">· 约 60 秒</span></div><h2>RVQ 更好、接近、K256 更好：三类都展示</h2><p><strong>这里的 RVQ 是预选新 5 点方案。</strong><a href="index.html#historical-gallery">原发版 10 点 RVQ 案例看这里</a>。下面是全部测试轨迹的分类占比；分类只比较终点偏差，卡片另给途中平均偏差。</p>
<div class="distribution" role="img" aria-label="RVQ占优60.9%，接近25.4%，K256占优13.7%">{distribution}</div><div class="legend"><span><i class="dot" style="background:{GREEN}"></i>RVQ 占优 {categories['rvq_better'] / preferred_test['routes']:.1%}</span><span><i class="dot" style="background:#9aa8bc"></i>接近 {categories['close'] / preferred_test['routes']:.1%}</span><span><i class="dot" style="background:{RED}"></i>K256 占优 {categories['k256_better'] / preferred_test['routes']:.1%}</span></div>
<p class="hero-note">“占优”要求终点偏差至少少 5 cm；其余归为接近。这是展示分组阈值，不是显著性门槛。每组展示 2 条，并不按组的总体比例配图。</p>
<div class="callout"><strong>看图方法：</strong>黑色＝GT 真实轨迹，红色＝K256，绿色＝新 RVQ。先看最终图里谁更贴近黑线，再点“播放轨迹动图”观察偏差怎样累积；箭头表示当前朝向。点击“看最终轨迹”可随时停止播放。</div>
{gallery_html}
<details><summary>如何选例、如何计点、坐标是否一致？</summary><p>固定保留先前选出的 6 条案例：每组在 GT 路程 5–10 m 与 10–25 m 各选 1 条，限制少于 140 帧，优先终点差接近候选中位数且来自不同场景。没有为本次视觉改版重选胜例。</p><p>每张图的三条路径共同平移、旋转，使共同起点朝向朝上；相同尺度、距离和误差保持不变。网格与右上角比例尺都以米计，不代表墙或 NavMesh。灰线是完整 GT，空心圆是 GT 终点，终态虚线连接 GT 与重建终点。</p><p>最终路径保留全部路点，含共同起点；每个 0.5 秒动作增加 1 点，STOP 不增加点。RVQ 每帧的 5 个局部前瞻点彼此重叠，动图画的是逐次执行首点得到的世界轨迹。播放最多抽取 75 帧，最终图保留全部点。</p></details>
</section>
<section class="recommendation" id="recommendation"><h2>现在可以怎么选</h2><ul><li><strong>需要保留未来轨迹：</strong>优先验证固定 0.5 秒、5 点、3 动作 token 的预选方案；本批平均终点偏差 {metric(preferred) * 100:.1f} cm，局部 5 点 ADE {metric(preferred,'local_ade_m') * 100:.2f} cm。</li><li><strong>只需要下一步：</strong>H1 可以更精确，但前瞻只剩 0.5 秒；需要减少规划请求时，独立验证整段执行。</li><li><strong>接入策略前：</strong>新 token 索引需要重新标注并训练 / 微调模型；停止由显式 stop 标志或 Affordance STOP 控制。新研究运动码本的 256 个索引都用于运动。</li></ul><details><summary>为什么不把测试终点更低的 14.2 cm 候选改称优选？</summary><p>额外约束首点位置的 H5 差分方案，本次测试终点均值为 {metric('new_h5_control_weighted') * 100:.1f} cm，但它在验证集的排序略落后。为了保留测试前的选择，首页仍用预先选定的绝对局部位姿方案；全部候选与三种子结果均在下方公开。后续可在实际任务的验证集上一起比较。</p></details></section>
<section id="evidence" class="appendix"><h2>完整实验与证据 <span class="reading-time">· 按需展开</span></h2><p>上述速读给出结论。这里保留全部 60 变体、16 份码本、失败实验、审计与复现材料。默认折叠以方便阅读。</p>{appendix}</section>
<p class="status-line">本次展示复核覆盖两页 21 组图表与 12 条轨迹，另提供 4 张手机竖排图。柱值、路点数、分类与轨迹误差均已核对；<a href="presentation-review.json">查看复核记录</a>。完整数值以 <a href="research-summary.json">机器可读结果</a> 为准；图中小数作了四舍五入。历史报告使用另一批 8,600 条轨迹，两页不混用样本。</p>
</main><script>
const filters=['family','tokens','search'].map(x=>document.getElementById(x));
function applyFilters(){{let n=0;for(const row of document.querySelectorAll('#all tbody tr')){{const show=(!filters[0].value||row.dataset.family===filters[0].value)&&(!filters[1].value||row.dataset.tokens===filters[1].value)&&row.dataset.name.toLowerCase().includes(filters[2].value.toLowerCase());row.hidden=!show;if(show)n++;}}document.getElementById('visible').textContent=`显示 ${{n}} / 60 个变体`;}}
for(const input of filters)input.addEventListener('input',applyFilters);applyFilters();
for(const button of document.querySelectorAll('[data-player]')){{button.addEventListener('click',()=>{{const image=document.getElementById(button.dataset.player);const playing=button.getAttribute('aria-pressed')!=='true';image.src=playing?button.dataset.gif:button.dataset.still;button.setAttribute('aria-pressed',String(playing));button.textContent=playing?'看最终轨迹':'播放轨迹动图';}});}}
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
