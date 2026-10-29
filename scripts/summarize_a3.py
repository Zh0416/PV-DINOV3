"""Verify retrieved A3 results and build paired group analysis and a local gallery."""

import argparse
import csv
from html import escape
import json
from pathlib import Path
import re
import statistics

from scripts.summarize_a2 import aggregate


def main(root, reference01, reference2):
    from PIL import Image
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    records, summary = read('per_image.json'), read('summary.json')
    queries = [r for r in records if r['included_in_summary']]
    refs = [{r['sample_id']: r for r in json.loads((directory / 'per_image.json').read_text(encoding='utf-8'))}
            for directory in (reference01, reference2)]
    names = list(summary['methods'])
    if len(records) != 60 or len(queries) != 58 or any(len(ref) != 60 for ref in refs):
        raise ValueError('Unexpected sample counts')
    for record in records:
        for previous in refs:
            old = previous[record['sample_id']]
            for field in ('included_in_summary', 'foreground_pixels', 'annotation', 'source_scene'):
                if old[field] != record[field]:
                    raise ValueError('Reference metadata mismatch: ' + record['sample_id'])
        for name in names:
            m = record['methods'][name]
            if sum(m[k] for k in ('tp', 'fp', 'fn', 'tn')) != 896 * 896:
                raise ValueError('Pixel total mismatch')
            if m['tp'] + m['fn'] != record['foreground_pixels']:
                raise ValueError('Foreground total mismatch')
            previous = refs[0] if name == 'A0' or name.startswith('A1_') else refs[1] if name.startswith('A2_') else None
            if previous and any(previous[record['sample_id']]['methods'][name][k] != m[k] for k in ('tp', 'fp', 'fn', 'tn')):
                raise ValueError('Prior control mismatch: ' + name)
        with Image.open(root / 'predictions' / (record['sample_id'] + '.png')) as image:
            if image.size != (1792, 296):
                raise ValueError('Unexpected panel shape')
            image.verify()
    for name in names:
        totals = aggregate(queries, name)
        for key in ('tp', 'fp', 'fn', 'tn', 'iou', 'precision', 'recall'):
            if totals[key] != summary['methods'][name][key]:
                raise ValueError('Summary mismatch: ' + name + '/' + key)
    for directory, methods in ((reference01, ['A0', 'A1_42', 'A1_43', 'A1_44']),
                               (reference2, ['A2_42', 'A2_43', 'A2_44'])):
        previous = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        for name in methods:
            if summary['methods'][name] != previous['methods'][name]:
                raise ValueError('Prior aggregate mismatch: ' + name)
    selection = read('fusion_selection.json')
    if selection['query_labels_used'] is not False:
        raise ValueError('Fusion selected using query labels')
    completion = read('completion.json')
    if completion['status'] != 'complete' or completion['gradient_updates'] != 0:
        raise ValueError('Run did not complete as a frozen diagnostic')
    for filename in ('selection_audit.json', 'data_observed.json'):
        if any(read(filename) != json.loads((directory / filename).read_text(encoding='utf-8'))
               for directory in (reference01, reference2)):
            raise ValueError('Reference role/data audit mismatch: ' + filename)
    thresholds = read('thresholds.json')
    if selection['calibration_ids'] != thresholds['calibration_ids']:
        raise ValueError('Calibration IDs inconsistent')
    if {r['sample_id'] for r in queries} & set(selection['calibration_ids']):
        raise ValueError('Query/calibration overlap')
    selected = max(selection['curve'], key=lambda r: (r['mean_calibration_iou'], -r['semantic_weight']))
    if selected['semantic_weight'] != summary['selected_semantic_weight']:
        raise ValueError('Fusion selected weight inconsistent')
    if completion['selected_semantic_weight'] != summary['selected_semantic_weight']:
        raise ValueError('Completion weight inconsistent')
    for seed in (42, 43, 44):
        if thresholds['methods']['A3_' + str(seed)] != selection['selected_runs'][str(seed)]:
            raise ValueError('Selected fusion thresholds inconsistent')
    for method in ('A1', 'A2', 'A3'):
        stored = summary['three_distinct_supports'][method]
        for key, value in stored['means'].items():
            values = [summary['methods'][method + '_' + str(seed)][key] for seed in (42, 43, 44)]
            if abs(statistics.mean(values) - value) > 1e-12:
                raise ValueError('Support mean mismatch')
        deviation = statistics.pstdev(summary['methods'][method + '_' + str(seed)]['iou'] for seed in (42, 43, 44))
        if abs(deviation - stored['micro_iou_population_std']) > 1e-12:
            raise ValueError('Support deviation mismatch')
    lookup = {r['sample_id']: r for r in records}
    with (root / 'per_image.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(records) * len(names):
        raise ValueError('CSV row count mismatch')
    if len({(r['sample_id'], r['method']) for r in csv_rows}) != len(csv_rows):
        raise ValueError('Duplicate CSV sample/method')
    for row in csv_rows:
        original = lookup[row['sample_id']]['methods'][row['method']]
        if any(int(row[k]) != original[k] for k in ('tp', 'fp', 'fn', 'tn')):
            raise ValueError('CSV confusion counts mismatch')
    groups = {}
    for field in ('area', 'difficulty', 'morphology'):
        groups[field] = {}
        for value in sorted({r['annotation'][field] for r in queries if r['foreground_pixels']}):
            members = [r for r in queries if r['foreground_pixels'] and r['annotation'][field] == value]
            data = {name: aggregate(members, name) for name in names}
            def mean_defined(method, key):
                values = [data[method + '_' + str(seed)][key] for seed in (42, 43, 44)]
                return statistics.mean(values) if all(v is not None for v in values) else None
            data['means'] = {method: {key: mean_defined(method, key)
                                      for key in ('iou', 'precision', 'recall', 'false_positive_rate')}
                             for method in ('A1', 'A2', 'A3')}
            data['count'] = len(members)
            groups[field][value or '(blank)'] = data
    positive = [r for r in queries if r['foreground_pixels']]
    negative = [r for r in queries if not r['foreground_pixels']]
    paired = {}
    for control in ('A1', 'A2'):
        paired[control] = {}
        for seed in (42, 43, 44):
            before, after = control + '_' + str(seed), 'A3_' + str(seed)
            paired[control][str(seed)] = {
                'positive_images_better': sum(r['methods'][after]['iou'] > r['methods'][before]['iou'] for r in positive),
                'positive_images_worse': sum(r['methods'][after]['iou'] < r['methods'][before]['iou'] for r in positive),
                'positive_images_tied': sum(r['methods'][after]['iou'] == r['methods'][before]['iou'] for r in positive),
                'negative_fp_area_decreased': sum(r['methods'][after]['fp'] < r['methods'][before]['fp'] for r in negative),
                'negative_fp_area_increased': sum(r['methods'][after]['fp'] > r['methods'][before]['fp'] for r in negative)}
    derived = {'groups_positive_only': groups, 'paired_A3_vs_controls': paired,
        'negative_images': {name: dict(aggregate(negative, name), images_with_any_fp=sum(r['methods'][name]['fp'] > 0 for r in negative)) for name in names},
        'source_scenes': {scene: {name: aggregate([r for r in queries if r['source_scene'] == scene], name) for name in names}
                          for scene in sorted({r['source_scene'] for r in queries})}}
    (root / 'analysis.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    percent = lambda x: f'{100 * x:.2f}%' if x is not None else '—'
    cards = []
    for record in records:
        anno, sid = record['annotation'], escape(record['sample_id'])
        label = '查询' if record['included_in_summary'] else '仅预览：不计入指标'
        attrs = {'id': record['sample_id'], 'area': anno['area'], 'difficulty': anno['difficulty'],
            'role': 'positive' if record['foreground_pixels'] else 'negative',
            'iou': record['methods']['A3_42']['iou'] or 0,
            'delta': (record['methods']['A3_42']['iou'] or 0) - (record['methods']['A2_42']['iou'] or 0)}
        attributes = ' '.join(f'data-{k}="{escape(str(v))}"' for k, v in attrs.items())
        scores = ' · '.join(f'{name}: {percent(record["methods"][name]["iou"])}' for name in ('A1_42', 'A2_42', 'A3_42'))
        false_alarms = ' · '.join(f'{name}误报占全图: {percent(record["methods"][name]["fp"] / 896**2)}' for name in ('A1_42', 'A2_42', 'A3_42'))
        cards.append(f'''<article {attributes}><h2>{sid} <small>{label}</small></h2>
<p>{escape(' · '.join(anno[k] for k in ('area', 'morphology', 'difficulty') if anno[k]))} · GT占比 {percent(record['foreground_pixels'] / 896**2)}</p>
<a href="predictions/{sid}.png"><img loading="lazy" src="predictions/{sid}.png" alt="{sid}八列预测对照"></a>
<p class="scores">前景IoU：{scores}<br>{false_alarms}</p></article>''')
    options = lambda field: ''.join(f'<option value="{escape(v)}">{escape(v)}</option>' for v in sorted({r['annotation'][field] for r in records if r['annotation'][field]}))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>A3 光伏融合实验</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#172237;margin:0}header,main{max-width:1800px;margin:auto;padding:22px}header{position:sticky;top:0;background:#f4f6faee;backdrop-filter:blur(8px);border-bottom:1px solid #ccd4df;z-index:2}h1{font-size:26px;margin:0}p{margin:8px 0}select,input{padding:8px;margin:4px;border:1px solid #ccd4df;border-radius:6px}article{background:white;padding:16px;margin:16px 0;border:1px solid #d8e0ea;border-radius:12px}h2{font-size:20px;margin:0}small{font-size:14px;font-weight:400;color:#67758a}img{width:100%;height:auto;display:block}.scores{font-variant-numeric:tabular-nums;color:#204f80}[hidden]{display:none}#count{margin-left:10px}</style>
<header><h1>A3：DINOv3 × GeoRSCLIP 空间对齐与固定融合</h1>
<p>八列：RGB、真值、A0响应、A1掩膜、A2响应／掩膜、A3响应／掩膜。掩膜展示run42；点击图片放大。</p>
<p>共享语义权重 ALPHA，仅用原12张校准图选择。58张查询、2张仅预览；3次独立1-shot，不集成。固定校准色标；响应不是概率。</p>
<input id="search" placeholder="输入影像ID"><select id="role"><option value="">全部类型</option><option value="positive">光伏正图</option><option value="negative">无光伏负图</option></select>
<select id="area"><option value="">全部面积</option>AREA</select><select id="difficulty"><option value="">全部背景</option>DIFFICULTY</select>
<select id="sort"><option value="id">按ID排序</option><option value="iou">A3 IoU从低到高</option><option value="delta">A3相对A2变化从低到高</option></select><span id="count"></span></header>
<main><p><a href="summary.json">汇总JSON</a> · <a href="per_image.csv">三次逐图CSV</a> · <a href="fusion_selection.json">校准曲线</a> · <a href="supports.png">支持图</a> · <a href="reference_reproduction.json">历史对照复核</a></p>CARDS</main>
<script>const cards=[...document.querySelectorAll('article')],main=document.querySelector('main');function update(){const value=id=>document.getElementById(id).value;let count=0;cards.forEach(c=>{c.hidden=!(c.dataset.id.includes(value('search').trim())&&['role','area','difficulty'].every(k=>!value(k)||c.dataset[k]===value(k)));count+=!c.hidden;});const order=value('sort');cards.sort((a,b)=>order==='id'?a.dataset.id.localeCompare(b.dataset.id):Number(a.dataset[order])-Number(b.dataset[order])).forEach(c=>main.append(c));document.getElementById('count').textContent=count+' / '+cards.length+' 张';}document.querySelectorAll('input,select').forEach(el=>el.addEventListener('input',update));update();</script></html>'''
    page = page.replace('ALPHA', str(summary['selected_semantic_weight'])).replace('AREA', options('area')).replace('DIFFICULTY', options('difficulty')).replace('CARDS', '\n'.join(cards))
    (root / 'index.html').write_text(page, encoding='utf-8')
    sources = re.findall(r'<img[^>]+src="([^"]+)"', page)
    if len(sources) != 60 or any(not (root / source).is_file() for source in sources):
        raise ValueError('Gallery image references invalid')
    integrity = {'status': 'passed', 'query_count': len(queries), 'preview_count': len(records) - len(queries),
        'decoded_png_count': len(sources), 'all_gallery_paths_valid': True,
        'all_confusion_pixel_totals_verified': True, 'pooled_summary_verified': True,
        'previous_A0_A1_A2_per_image_counts_and_aggregates_identical': True,
        'calibration_only_weight_selection_verified': True,
        'support_means_and_deviation_verified': True,
        'csv_count_and_confusions_verified': True}
    (root / 'result_integrity.json').write_text(json.dumps(integrity, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'integrity': integrity, 'supports': summary['three_distinct_supports'],
                      'weight': summary['selected_semantic_weight'], 'paired': paired,
                      'groups': {f: {v: dict(count=d['count'], means=d['means']) for v, d in values.items()}
                                 for f, values in groups.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference01', type=Path, required=True)
    parser.add_argument('--reference2', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reference01, args.reference2)
