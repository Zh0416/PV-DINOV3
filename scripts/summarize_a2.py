"""Audit retrieved A2 reports, summarize paired groups, and build a local gallery."""

import argparse
from html import escape
import json
from pathlib import Path
import re
import statistics


def aggregate(rows, name):
    counts = {key: sum(r['methods'][name][key] for r in rows) for key in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn, tn = [counts[k] for k in ('tp', 'fp', 'fn', 'tn')]
    divide = lambda a, b: a / b if b else None
    return dict(counts, count=len(rows), iou=divide(tp, tp + fp + fn),
        f1=divide(2 * tp, 2 * tp + fp + fn), precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn), false_positive_rate=divide(fp, fp + tn))


def main(root, reference):
    from PIL import Image
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    records, summary = read('per_image.json'), read('summary.json')
    queries = [r for r in records if r['included_in_summary']]
    previous = json.loads((reference / 'per_image.json').read_text(encoding='utf-8'))
    previous = {r['sample_id']: r for r in previous}
    names = list(summary['methods'])
    if len(records) != 60 or len(queries) != 58 or len(previous) != 60:
        raise ValueError('Unexpected selected sample counts')
    for record in records:
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
            if name.startswith('A1_') and any(old['methods'][name][k] != m[k] for k in ('tp', 'fp', 'fn', 'tn')):
                raise ValueError('Paired A1 mismatch')
        path = root / 'predictions' / (record['sample_id'] + '.png')
        with Image.open(path) as image:
            if image.size != (1344, 296):
                raise ValueError('Unexpected panel shape')
            image.verify()
    for name in names:
        totals = aggregate(queries, name)
        for key in ('tp', 'fp', 'fn', 'tn', 'iou', 'precision', 'recall'):
            if totals[key] != summary['methods'][name][key]:
                raise ValueError('Summary mismatch: ' + name + '/' + key)
    groups = {}
    for field in ('area', 'difficulty', 'morphology'):
        groups[field] = {}
        for value in sorted({r['annotation'][field] for r in queries if r['foreground_pixels']}):
            members = [r for r in queries if r['foreground_pixels'] and r['annotation'][field] == value]
            data = {name: aggregate(members, name) for name in names}
            def mean_if_defined(method, key):
                values = [data[method + '_' + str(seed)][key] for seed in (42, 43, 44)]
                return statistics.mean(values) if all(v is not None for v in values) else None
            data['means'] = {method: {k: mean_if_defined(method, k)
                for k in ('iou', 'precision', 'recall')} for method in ('A1', 'A2')}
            data['count'] = len(members)
            groups[field][value or '(blank)'] = data
    positive = [r for r in queries if r['foreground_pixels']]
    negatives = [r for r in queries if not r['foreground_pixels']]
    paired = {}
    for seed in (42, 43, 44):
        a1, a2 = 'A1_' + str(seed), 'A2_' + str(seed)
        paired[str(seed)] = {
            'positive_images_better': sum(r['methods'][a2]['iou'] > r['methods'][a1]['iou'] for r in positive),
            'positive_images_worse': sum(r['methods'][a2]['iou'] < r['methods'][a1]['iou'] for r in positive),
            'positive_images_tied': sum(r['methods'][a2]['iou'] == r['methods'][a1]['iou'] for r in positive),
            'negative_fp_area_decreased': sum(r['methods'][a2]['fp'] < r['methods'][a1]['fp'] for r in negatives),
            'negative_fp_area_increased': sum(r['methods'][a2]['fp'] > r['methods'][a1]['fp'] for r in negatives)}
    derived = {'groups_positive_only': groups, 'paired': paired,
        'negative_images': {name: dict(aggregate(negatives, name),
            images_with_any_fp=sum(r['methods'][name]['fp'] > 0 for r in negatives)) for name in names},
        'source_scenes': {scene: {name: aggregate([r for r in queries if r['source_scene'] == scene], name) for name in names}
                          for scene in sorted({r['source_scene'] for r in queries})}}
    (root / 'analysis.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    percent = lambda x: f'{100 * x:.2f}%' if x is not None else '—'
    cards = []
    for record in records:
        anno = record['annotation']
        label = '查询' if record['included_in_summary'] else '仅预览：不计入指标'
        attrs = {'id': record['sample_id'], 'area': anno['area'], 'difficulty': anno['difficulty'],
            'role': 'positive' if record['foreground_pixels'] else 'negative',
            'iou': record['methods']['A2_42']['iou'] or 0,
            'delta': (record['methods']['A2_42']['iou'] or 0) - (record['methods']['A1_42']['iou'] or 0)}
        attributes = ' '.join(f'data-{k}="{escape(str(v))}"' for k, v in attrs.items())
        scores = ' · '.join(f'{name}: {percent(record["methods"][name]["iou"])}' for name in names)
        sid = escape(record['sample_id'])
        cards.append(f'''<article {attributes}><h2>{sid} <small>{label}</small></h2>
<p>{escape(' · '.join(anno[k] for k in ('area', 'morphology', 'difficulty') if anno[k]))} · GT占比 {percent(record['foreground_pixels'] / (896 * 896))}</p>
<a href="predictions/{sid}.png"><img loading="lazy" src="predictions/{sid}.png" alt="{sid}六列预测对照"></a>
<p class="scores">前景IoU：{scores}</p></article>''')
    options = lambda field: ''.join(f'<option value="{escape(v)}">{escape(v)}</option>' for v in sorted({r['annotation'][field] for r in records if r['annotation'][field]}))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>A1 / A2 光伏实验预测</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#172237;margin:0}header,main{max-width:1500px;margin:auto;padding:22px}header{position:sticky;top:0;background:#f4f6faee;backdrop-filter:blur(8px);border-bottom:1px solid #ccd4df;z-index:2}h1{font-size:26px;margin:0}p{margin:8px 0}select,input{padding:8px;margin:4px;border:1px solid #ccd4df;border-radius:6px}article{background:white;padding:16px;margin:16px 0;border:1px solid #d8e0ea;border-radius:12px}h2{font-size:20px;margin:0}small{font-size:14px;font-weight:400;color:#67758a}img{width:100%;height:auto;display:block}.scores{font-variant-numeric:tabular-nums;color:#204f80}[hidden]{display:none}#count{margin-left:10px}</style>
<header><h1>A1 / A2 前景与背景原型对照</h1><p>六列：RGB、真值、A1响应、A1掩膜、A2响应、A2掩膜，均展示支持图pv001183（run42）。点击图片可放大。</p>
<p>58张查询、2张仅预览；三个支持各自独立1-shot，不集成。热图分别使用校准集固定色标，颜色不等于概率。</p>
<input id="search" placeholder="输入影像ID"><select id="role"><option value="">全部类型</option><option value="positive">光伏正图</option><option value="negative">无光伏负图</option></select>
<select id="area"><option value="">全部面积</option>AREA</select><select id="difficulty"><option value="">全部背景</option>DIFFICULTY</select>
<select id="sort"><option value="id">按ID排序</option><option value="iou">A2 IoU从低到高</option><option value="delta">A2相对A1变化从低到高</option></select><span id="count"></span></header>
<main><p><a href="summary.json">汇总JSON</a> · <a href="per_image.csv">逐图CSV</a> · <a href="supports.png">支持图</a> · <a href="reference_reproduction.json">A1复核</a></p>CARDS</main>
<script>const cards=[...document.querySelectorAll('article')],main=document.querySelector('main');
function update(){const value=id=>document.getElementById(id).value;let count=0;cards.forEach(c=>{c.hidden=!(c.dataset.id.includes(value('search').trim())&&['role','area','difficulty'].every(k=>!value(k)||c.dataset[k]===value(k)));count+=!c.hidden;});const order=value('sort');cards.sort((a,b)=>order==='id'?a.dataset.id.localeCompare(b.dataset.id):Number(a.dataset[order])-Number(b.dataset[order])).forEach(c=>main.append(c));document.getElementById('count').textContent=count+' / '+cards.length+' 张';}document.querySelectorAll('input,select').forEach(el=>el.addEventListener('input',update));update();</script></html>'''
    page = page.replace('AREA', options('area')).replace('DIFFICULTY', options('difficulty')).replace('CARDS', '\n'.join(cards))
    (root / 'index.html').write_text(page, encoding='utf-8')
    sources = re.findall(r'<img[^>]+src="([^"]+)"', page)
    if len(sources) != 60 or any(not (root / source).is_file() for source in sources):
        raise ValueError('Gallery image references invalid')
    integrity = {'status': 'passed', 'query_count': len(queries), 'preview_count': len(records) - len(queries),
        'decoded_png_count': len(sources), 'all_gallery_paths_valid': True,
        'all_confusion_pixel_totals_verified': True, 'pooled_summary_verified': True,
        'previous_A1_per_image_counts_identical': True}
    (root / 'result_integrity.json').write_text(json.dumps(integrity, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'integrity': integrity, 'supports': summary['three_distinct_supports'],
                      'paired': paired, 'groups': {f: {v: dict(count=d['count'], means=d['means']) for v, d in values.items()}
                                                   for f, values in groups.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reference)
