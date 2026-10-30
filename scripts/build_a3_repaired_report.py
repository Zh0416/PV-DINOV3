"""Audit downloaded derived reports and build a single offline Chinese HTML."""
import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pvseg.frozen_baselines import metrics

SEEDS = (42, 43, 44)
KEYS = ('tp', 'fp', 'fn', 'tn')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pooled(rows, method):
    return metrics({k: sum(r['methods'][method][k] for r in rows) for k in KEYS})


def mean_metric(rows, method, key='iou'):
    if key == 'negative_pixel_false_positive_rate':
        rows = [r for r in rows if not r['foreground_pixels']]
        key = 'false_positive_rate'
    values = [pooled(rows, f'{method}_{seed}')[key] for seed in SEEDS]
    return statistics.mean(values) if all(v is not None for v in values) else None


def pct(v):
    return '—' if v is None else f'{100 * v:.2f}%'


def delta(a, b):
    return '—' if a is None or b is None else f'{100 * (a-b):+.2f} pp'


def data_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def audit(root, old):
    completion, summary = read(root / 'completion.json'), read(root / 'summary.json')
    rows = read(root / 'per_image.json')
    included = [r for r in rows if r['included_in_summary']]
    assert completion['status'] == 'complete' and completion['episode_updates'] == 320
    assert completion['exact_A2_reproduction'] and not completion['query_labels_used_for_selection']
    assert len(rows) == 60 and len(included) == 58 and len({r['sample_id'] for r in rows}) == 60
    assert read(root / 'training_allocation.json') == read(old / 'training_allocation.json')
    reuse = read(root / 'base_reuse.json')
    assert reuse['allocation_identical'] and reuse['schedule_identical'] and reuse['step'] == 160
    locked = read(root / 'selection_locked.json')
    assert not locked['query_images_loaded'] and not locked['query_labels_loaded']
    history = read(root / 'episode_training.json')
    assert len(history) == 320 and [x['step'] for x in history] == list(range(1, 321))
    schedule = read(root / 'episode_schedule.json')
    assert all((r['query'], r['support']) == (s['query'], s['support']) for r, s in zip(history, schedule))
    assert all(math.isfinite(r['loss']) and all(math.isfinite(v) for v in r['terms'].values()) for r in history)
    assert all('balanced_meta_ce' in r['terms'] for r in history)
    curve = read(root / 'checkpoint_selection.json')['curve']
    assert [x['step'] for x in curve] == [80, 160, 240, 320]
    for x in curve:
        expected = statistics.mean(metrics(x['fixed_final_counts'][str(seed)])['iou'] for seed in SEEDS)
        assert abs(expected - x['mean_calibration_iou']) < 1e-12
    selected = max(curve, key=lambda x: (x['mean_calibration_iou'], -x['step']))
    assert selected['step'] == summary['selected_step'] == locked['selected_step'] == completion['selected_step']
    before = {r['sample_id']: r for r in read(old / 'per_image.json')}
    for r in rows:
        sid = r['sample_id']
        assert (root / 'predictions' / (sid + '.png')).is_file()
        assert r['included_in_summary'] == before[sid]['included_in_summary']
        assert r['annotation'] == before[sid]['annotation']
        for name, result in r['methods'].items():
            assert all(isinstance(result[k], int) and result[k] >= 0 for k in KEYS)
            assert sum(result[k] for k in KEYS) == 896 * 896
            assert result['tp'] + result['fn'] == r['foreground_pixels']
            for k, v in metrics({k: result[k] for k in KEYS}).items():
                assert v == result[k], (sid, name, k)
        for seed in SEEDS:
            assert all(r['methods'][f'A2_{seed}'][k] == before[sid]['methods'][f'A2_{seed}'][k] for k in KEYS)
    for method, result in summary['methods'].items():
        for k, v in pooled(included, method).items():
            assert result[k] == v, (method, k)
        negatives = [r for r in included if not r['foreground_pixels']]
        assert result['negative_images_with_any_fp'] == sum(r['methods'][method]['fp'] > 0 for r in negatives)
    for field, groups in summary['groups'].items():
        for value, methods in groups.items():
            members = [r for r in included if r['annotation'][field] == value]
            for name, result in methods.items():
                assert all(result[k] == v for k, v in pooled(members, name).items())
    for method, result in summary['means_over_support_choices'].items():
        for k, v in result.items():
            recomputed = mean_metric(included, method, k)
            assert (v is None and recomputed is None) or abs(v - recomputed) < 1e-12
    test_text = (root / 'tests.txt').read_text()
    assert 'Ran 39 tests' in test_text and '\nOK' in test_text
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.json')) if p.name != 'report_integrity.json'}
    report = dict(status='passed', statistical_queries=58, previews=2, support_choices=3,
        successful_updates=320, selected_step=selected['step'], exact_historical_A2_counts=True,
        checkpoint_selection_recomputed=True, per_image_and_group_counts_recomputed=True,
        finite_all_losses=True, reports_sha256=manifest)
    (root / 'report_integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    return rows, included, summary, history, curve, report


def chart(root, history, curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
    x = [r['step'] for r in history]
    for term, label in [('total', 'Total (new objective)'), ('balanced_meta_ce', 'Balanced raw-meta CE')]:
        axes[0].plot(x, [r['terms'][term] for r in history], label=label, alpha=.8, linewidth=1)
    axes[0].set(xlabel='Stage-two update', ylabel='Loss (log scale)', yscale='log')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.2)
    axes[1].plot([r['step'] for r in curve], [100*r['mean_calibration_iou'] for r in curve], 'o-', color='#006b62')
    best = max(curve, key=lambda r: (r['mean_calibration_iou'], -r['step']))
    axes[1].scatter([best['step']], [100*best['mean_calibration_iou']], s=100, facecolors='none', edgecolors='#e39824', linewidths=2)
    axes[1].set(xlabel='Checkpoint update', ylabel='Calibration full-output IoU (%)', xticks=[80, 160, 240, 320])
    axes[1].grid(alpha=.2)
    path = root / 'training_selection.png'
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)
    return path


def build(root, old, output):
    rows, included, summary, history, curve, integrity = audit(root, old)
    old_rows = [r for r in read(old / 'per_image.json') if r['included_in_summary']]
    final = mean_metric(included, 'final_fixed')
    calibrated = mean_metric(included, 'final')
    a2 = mean_metric(included, 'A2')
    base = mean_metric(included, 'base_fixed')
    old_final = mean_metric(old_rows, 'final')
    if final == 0:
        result = '完整输出仍未通过有效前景检出检查'
    elif final < a2:
        result = '完整输出检出了前景，但整体 IoU 仍低于 A2'
    else:
        result = '完整输出检出了前景，本轮开发集 IoU 超过 A2'
    versus_a2 = f'{"高" if final >= a2 else "低"} {100 * abs(final-a2):.2f} 个百分点'
    versus_base = f'{"高" if final >= base else "低"} {100 * abs(final-base):.2f} 个百分点'
    comparison = f'固定阈值完整输出的 IoU 较 A2 {versus_a2}，较冻结 CNN {versus_base}。'
    interpretation = (f'固定阈值完整输出的召回率为 {pct(mean_metric(included, "final_fixed", "recall"))}，'
        f'漏检仍是主要限制；校准阈值后 IoU 为 {pct(calibrated)}。融合先验 IoU 为 '
        f'{pct(mean_metric(included, "fused"))}，而 meta 输出在固定阈值下为 '
        f'{pct(mean_metric(included, "meta_fixed"))}，与 final 的 {pct(final)} 接近。'
        '性能差距在 meta 分支就已出现，需要优先检查先验到解码器的利用、跨场景训练覆盖及基础分支适配。'
        f'冻结 CNN 的 IoU 仅 {pct(base)}，也是需要处理的独立问题；这些现象尚不足以确定唯一根因。')
    e = html.escape
    table_rows = []
    methods = [('A2', 'A2：冻结 DINOv3 前景/背景原型', '历史校准阈值'),
               ('base_fixed', '冻结 PSPNet / ResNet-50', '0.5'),
               ('meta_fixed', '修正后 meta 输出', '0.5'),
               ('final_fixed', '修正后完整输出（主结果）', '0.5'),
               ('final', '修正后完整输出（校准阈值）', '独立校准集'),
               ('fused', '修正后融合先验 CAM（诊断）', '独立校准集')]
    for method, label, threshold in methods:
        table_rows.append('<tr' + (' class="primary"' if method == 'final_fixed' else '') + '><td>' + label + '</td><td>' + threshold + '</td>' + ''.join('<td>' + pct(mean_metric(included, method, key)) + '</td>' for key in ('iou', 'precision', 'recall', 'false_positive_rate')) + '</tr>')
    for method, label in [('final_fixed', '旧 learned A3 完整输出（0.5）'), ('final', '旧 learned A3 完整输出（校准）'), ('fused', '旧 learned A3 融合先验（诊断）')]:
        table_rows.append('<tr class="old"><td>' + label + '</td><td>历史规则</td>' + ''.join('<td>' + pct(mean_metric(old_rows, method, key)) + '</td>' for key in ('iou', 'precision', 'recall', 'false_positive_rate')) + '</tr>')
    group_rows = []
    for value in sorted({r['annotation']['difficulty'] for r in included}):
        members = [r for r in included if r['annotation']['difficulty'] == value]
        previous = [r for r in old_rows if r['annotation']['difficulty'] == value]
        group_rows.append('<tr><td>' + e(value) + '</td><td>' + str(len(members)) + '</td>' + ''.join('<td>' + pct(v) + '</td>' for v in (mean_metric(members, 'A2'), mean_metric(members, 'base_fixed'), mean_metric(members, 'final_fixed'), mean_metric(members, 'final'), mean_metric(previous, 'final'))) + '<td>' + delta(mean_metric(members, 'final_fixed'), mean_metric(members, 'A2')) + '</td></tr>')
    positive_group_rows = []
    for value in ('大棚干扰', '建筑复杂'):
        members = [r for r in included if r['annotation']['difficulty'] == value and r['foreground_pixels'] > 0]
        previous = [r for r in old_rows if r['annotation']['difficulty'] == value and r['foreground_pixels'] > 0]
        positive_group_rows.append('<tr><td>' + value + '</td><td>' + str(len(members)) + '</td>' + ''.join('<td>' + pct(v) + '</td>' for v in (mean_metric(members, 'A2'), mean_metric(members, 'final_fixed'), mean_metric(members, 'final'), mean_metric(previous, 'final'))) + '</tr>')
    negatives = [r for r in included if r['foreground_pixels'] == 0]
    neg_rows = []
    support_rows = []
    for seed in SEEDS:
        for method, label in [('A2', 'A2'), ('base_fixed', '冻结 CNN'), ('final_fixed', '完整输出 0.5'), ('final', '完整输出校准')]:
            m = f'{method}_{seed}'
            n, total = pooled(negatives, m), pooled(included, m)
            neg_rows.append(f'<tr><td>{seed}</td><td>{label}</td><td>{n["fp"]:,}</td><td>{pct(n["false_positive_rate"])}</td><td>{sum(r["methods"][m]["fp"] > 0 for r in negatives)} / {len(negatives)}</td></tr>')
            support_rows.append(f'<tr><td>{seed}</td><td>{label}</td><td>{pct(total["iou"])}</td><td>{pct(total["precision"])}</td><td>{pct(total["recall"])}</td><td>{total["tp"]:,}</td><td>{total["fp"]:,}</td><td>{total["fn"]:,}</td></tr>')
    difficulties = sorted({r['annotation']['difficulty'] for r in rows})
    samples = []
    for r in rows:
        r = dict(r)
        r['image'] = data_uri(root / 'predictions' / (r['sample_id'] + '.png'))
        samples.append(r)
    threshold = read(root / 'thresholds.json')
    env = read(root / 'environment.json')
    reuse = read(root / 'base_reuse.json')
    selection_rows = ''.join(f'<tr><td>{r["step"]}</td><td>{pct(r["mean_calibration_iou"])}</td><td>{"选中" if r["step"] == summary["selected_step"] else ""}</td></tr>' for r in curve)
    payload = json.dumps(samples, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    content = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SSPNet A3｜损失修正后的完整评估</title>
<style>
:root{--ink:#163338;--muted:#536970;--accent:#00776b;--line:#d7e3e0;--paper:#fff;--bg:#f2f6f4}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1240px;margin:auto;padding:34px 26px 70px}h1{font-size:34px;line-height:1.35;margin:12px 0}h2{font-size:24px;margin:0 0 16px}h3{font-size:18px;margin:20px 0 8px}p{margin:10px 0}a{color:var(--accent)}.eyebrow{font-size:13px;letter-spacing:.08em;color:var(--accent)}.muted,small{color:var(--muted)}section{background:var(--paper);border:1px solid var(--line);border-radius:15px;padding:26px;margin-top:22px}.hero{background:#e8f3ee}.lead{font-size:20px;font-weight:600}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.metric{background:#fff;border:1px solid var(--line);border-radius:10px;padding:17px}.metric strong{display:block;font-size:30px;color:var(--accent)}.metric span{font-size:13px}.note{border-left:4px solid #d59a38;background:#fff9e9;padding:12px 16px;margin:16px 0}.scroll{overflow:auto}table{border-collapse:collapse;min-width:700px;width:100%;font-size:14px;white-space:nowrap}th,td{text-align:right;padding:11px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{background:#f1f6f4;font-weight:600}.primary{background:#e7f6ed;font-weight:600}.old{color:var(--muted)}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.box{padding:16px;border-radius:10px;border:1px solid var(--line);background:#f7faf8}.box b{display:block}.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.filters label{font-size:14px}input,select,button{font:inherit;border:1px solid #a9c3b9;border-radius:7px;background:white;padding:7px 10px;color:var(--ink)}input{width:180px}button{cursor:pointer;background:#00776b;color:#fff}button:disabled{opacity:.4;cursor:default}.sample{margin:22px 0;border-top:1px solid var(--line);padding-top:16px}.sample img{display:block;width:100%;max-width:1008px;height:auto;margin:14px auto;border:1px solid var(--line)}.sample h3{margin:0}.tag{font-size:12px;background:#eef4ef;border-radius:5px;padding:3px 8px;margin-left:6px}.pagination{display:flex;gap:14px;justify-content:center;align-items:center;margin:22px 0}.chart{width:100%;height:auto}details{margin:15px 0}summary{cursor:pointer;color:var(--accent)}pre{font-size:12px;background:#f1f5f2;padding:15px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}footer{font-size:13px;margin:28px 5px;color:var(--muted)}li{margin:6px 0}@media(max-width:800px){main{padding:16px 12px}.grid{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr}section{padding:18px}h1{font-size:27px}.metric strong{font-size:25px}}@media print{body{background:white}.filters,.pagination{display:none}section{break-inside:avoid}.sample{break-inside:avoid}main{max-width:none}}</style></head><body><main>
<header><div class="eyebrow">SSPNET × DINOv3-SAT493M · 2026-10-30 · A3 后续实验</div>
<h1>损失修正后的完整网络评估</h1><p class="muted">原生 896 × 896 光伏切片 · 训练侧修正后，回到原划分验证 · 单文件离线报告</p>
</header>
<section class="hero"><div class="eyebrow">本轮结论</div>
<p class="lead">@@RESULT@@</p>
<p>@@COMPARISON@@</p>
<div class="grid"><div class="metric"><span>完整输出 · 固定阈值 0.5</span><strong>@@FINAL@@</strong><span>前景 micro-IoU</span></div>
<div class="metric"><span>完整输出 · 校准阈值</span><strong>@@CAL@@</strong><span>同一检查点的另一输出规则</span></div>
<div class="metric"><span>冻结 A2</span><strong>@@A2@@</strong><span>逐图计数与历史结果一致</span></div>
<div class="metric"><span>冻结 CNN 基础分支</span><strong>@@BASE@@</strong><span>原 step160 权重复用</span></div>
</div>
<p>统计对象为 58 张查询图；另 2 张仅作预览。每次只使用 1 张支持图，分别更换 3 张保留支持，指标先汇总每次的像素计数，再对 3 次结果求均值。</p>
<div class="note">这是额外像素标注监督下的已知光伏类别开发评估。当前查询集已多次查看，且训练仅来自一个空间组；它不能替代独立空间测试，也不能说明原论文的新类别小样本基准已复现。</div>
</section>

<section><h2>这一步改了什么</h2>
<p>上一轮同初始化诊断显示，meta logits 放大后 softmax 饱和，原有损失的纠错梯度消失；仅切换 FP32 仍全背景。新增直接作用于原始 meta logits 的类别均衡交叉熵，已在固定训练侧诊断中恢复前景。本轮将这一修正带回完整的 22 张训练图，重新初始化并训练第二阶段。</p>
<div class="flow"><div class="box"><b>① 冻结编码器</b>查询 RGB + 支持 RGB / mask → DINOv3 1024 维特征、PSPNet 多层特征。GeoRSCLIP 提供冻结的光伏文本向量。</div>
<div class="box"><b>② 可学习先验与交互</b>DMSF 共享视觉投影、文本 MLP、互补 CAM 与可学习融合 → 先验组装、DCMA、ASPP → meta 分割。</div>
<div class="box"><b>③ 完整分割与损失修正</b>meta / base 合并 → 896 × 896 前景分数与 mask。原四项损失 + 1.0 × 类别均衡 raw-meta CE。</div>
</div>
<p>均衡 CE 在每个训练 episode 内分别计算实际存在类别的像素平均损失，再对类别等权平均；纯负样本只监督背景。它是此次适配修正，不属于论文原始损失。</p>
<ul><li>原 22 张训练图（4 正、18 负），原 320 次查询/支持顺序，AdamW 1e-4，梯度裁剪 10，无增强。</li>
<li>复用旧 A3 step160 的 PSPNet；第二阶段从 seed42 新初始化。训练和推理均使用 FP32。</li>
<li>每 80 次更新在 12 张校准图上评估；按完整输出在固定 0.5 阈值下的三支持均值 IoU 选检查点，并列取较早步骤。选定后才打开查询图像和标签像素。</li>
</ul><p class="muted">与旧 learned A3 相比，本轮同时改变了损失、精度和检查点选择规则，因此完整结果对比不是单因素消融；损失作用的专门证据来自上一轮配对诊断。</p>
</section>

<section><h2>整体结果</h2>
<div class="scroll"><table><thead><tr><th>方法 / 输出</th><th>阈值</th><th>前景 IoU ↑</th><th>Precision ↑</th><th>Recall ↑</th><th>背景像素 FPR ↓</th></tr></thead>
<tbody>@@TABLE@@</tbody>
</table></div>
<p class="muted">IoU = TP / (TP + FP + FN)；FPR = FP / (FP + TN)。融合先验 CAM 是中间诊断量，不能视为完整网络分割成绩。旧 A3 使用先验成绩选择检查点。新模型还使用了 A2 未使用的 22 张训练 mask，二者的差异不能单独归因于融合模块。</p>
<details><summary>查看 3 种支持各自的原始计数与指标</summary><div class="scroll"><table><thead><tr><th>支持选择</th><th>方法</th><th>IoU</th><th>Precision</th><th>Recall</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
<tbody>@@SUPPORT@@</tbody>
</table></div>
<p>42 / 43 / 44 是历史支持选择编号，不是 3 次独立训练。</p>
</details></section>

<section><h2>大棚、建筑与其他困难组</h2>
<div class="scroll"><table><thead><tr><th>用户标注困难组</th><th>图数</th><th>A2 IoU</th><th>冻结 CNN</th><th>新完整 0.5</th><th>新完整校准</th><th>旧完整校准</th><th>新 0.5 − A2</th></tr></thead>
<tbody>@@GROUPS@@</tbody>
</table></div>
<p class="muted">全部分组由用户原始选片标注确定。纯负组没有真实前景，IoU 无法评价检出能力，请看下表误检率。各组图数较少，不作显著性或泛化结论。</p>
<h3>纯负样本的误检</h3><div class="scroll"><table><thead><tr><th>支持选择</th><th>方法</th><th>误检像素数</th><th>负图像素 FPR ↓</th><th>出现任意误检的负图数 ↓</th></tr></thead>
<tbody>@@NEG@@</tbody>
</table></div>
</section>

<section><h2>训练与检查点选择</h2>
<img class="chart" src="@@CHART@@" alt="真实训练损失与校准完整输出 IoU"><div class="scroll"><table><thead><tr><th>更新步骤</th><th>校准完整输出固定 0.5 IoU</th><th>选择</th></tr></thead>
<tbody>@@SELECTION@@</tbody>
</table></div>
<p>完成 320 / 320 次更新；选中 step @@STEP@@。查询集不参与检查点或阈值选择。左图仅展示本轮新损失，其绝对值不能与项数不同的旧总损失直接比较。</p>
<p>本地与 Kaggle 的 39 项测试通过；离线再次复算逐图、分组和总指标，并验证全部 60 图的 A2 计数与历史一致。代码和本次流程通过这些检查，不代表没有潜在错误或已经达到论文精度。</p>
</section>

<section id="gallery"><h2>逐图检查 · 全部 60 张</h2>
<p>每个面板从左到右、从上到下：RGB、真实标签、A2；冻结 CNN、meta 0.5、完整输出 0.5；完整输出校准、0.5 输出的错误图、融合先验。错误图绿色为 TP、红色为 FP、蓝色为 FN；先验颜色范围固定 [0,1]，不是概率。</p>
<p class="muted">图片使用同一个历史支持 pv001183（编号 42）；下方表格可切换 3 种支持，图片不会随数值选择变化。所有面板为原生推理结果缩小预览，点击可放大查看。</p>
<div class="filters"><label>困难组 <select id="difficulty"><option value="">全部</option>@@OPTIONS@@</select></label><label>范围 <select id="role"><option value="">全部 60 张</option><option value="query">统计查询 58 张</option><option value="preview">仅预览 2 张</option><option value="negative">纯负样本</option></select></label><label>样本 <input id="search" placeholder="输入 pv 编号"></label><label>数值支持 <select id="seed"><option>42</option><option>43</option><option>44</option></select></label></div>
<div class="pagination"><button id="prev">上一页</button><span id="count"></span><button id="next">下一页</button></div>
<div id="samples"></div>
</section>

<section><h2>如何判断这一步是否成功</h2>
<p>工程门槛是完整分割输出能否持续产生有效前景；效果门槛是查询图上的 IoU 和误检能否优于 A2 与冻结 CNN。两者应分别判断，不能只看训练损失下降或融合 CAM 变好。</p>
<p>@@COMPARISON@@ 后续应先根据大棚、建筑和纯负图的 FP / FN 分布确定主要失败类型，再安排有独立空间验证的数据扩充与单因素消融。比较损失或融合结构时，固定精度、选择规则和训练顺序；在扩大训练前保留一套尚未查看的空间测试集。</p>
<details><summary>可追溯配置、阈值与本地审计</summary><pre>@@EVIDENCE@@</pre></details><p><a href="https://www.kaggle.com/code/zh0416/pv-a3-repaired-full-evaluation">Kaggle 实验</a> · <a href="https://github.com/Zh0416/PV-DINOV3/tree/@@COMMIT@@">固定源码版本</a></p>
</section>

<footer>SSPNet 方法重建与 DINOv3 光伏适配 · 报告基于实际完成的 Kaggle 作业。训练原图、基础权重和检查点保留在 Kaggle；本文件包含派生预览与指标，可离线打开。</footer></main>
<dialog id="zoom" style="max-width:98vw;max-height:96vh;padding:8px;border:0;border-radius:10px"><button id="closeZoom">关闭</button><img id="zoomImage" style="display:block;max-width:92vw;max-height:85vh;object-fit:contain" alt="放大预测面板"></dialog>
<script id="sampleData" type="application/json">@@DATA@@</script><script>
const data=JSON.parse(document.getElementById('sampleData').textContent);let page=0;const size=4;
const $=id=>document.getElementById(id);const percent=v=>v===null?'—':(100*v).toFixed(2)+'%';
function render(){const d=$('difficulty').value,r=$('role').value,q=$('search').value.trim().toLowerCase(),seed=$('seed').value;
const filtered=data.filter(x=>(!d||x.annotation.difficulty===d)&&(!q||x.sample_id.toLowerCase().includes(q))&&(!r||r==='query'&&x.included_in_summary||r==='preview'&&!x.included_in_summary||r==='negative'&&x.foreground_pixels===0));
const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);$('count').textContent=`${filtered.length} 张 · 第 ${page+1} / ${pages} 页`;$('prev').disabled=page===0;$('next').disabled=page>=pages-1;
$('samples').innerHTML='';for(const x of filtered.slice(page*size,(page+1)*size)){const el=document.createElement('article');el.className='sample';const title=document.createElement('h3');title.textContent=x.sample_id+' · '+x.annotation.difficulty+' · '+(x.included_in_summary?'统计查询':'仅预览，不计入指标');el.append(title);const text=document.createElement('p');text.className='muted';text.textContent=`${x.annotation.area} / ${x.annotation.morphology} · 标签前景 ${x.foreground_pixels.toLocaleString()} 像素 · ${x.source_scene}`;el.append(text);
const image=document.createElement('img');image.src=x.image;image.alt=x.sample_id+' 预测对比';image.loading='lazy';image.style.cursor='zoom-in';image.onclick=()=>{$('zoomImage').src=x.image;$('zoom').showModal()};el.append(image);
const wrap=document.createElement('div');wrap.className='scroll';const table=document.createElement('table');table.innerHTML='<thead><tr><th>支持 '+seed+' · 方法</th><th>IoU</th><th>Precision</th><th>Recall</th><th>FP</th><th>FN</th></tr></thead>';const body=document.createElement('tbody');for(const [m,label] of [['A2','A2'],['base_fixed','冻结 CNN'],['meta_fixed','Meta 0.5'],['final_fixed','完整 0.5'],['final','完整校准']]){const v=x.methods[m+'_'+seed];const row=document.createElement('tr');for(const value of [label,percent(v.iou),percent(v.precision),percent(v.recall),v.fp.toLocaleString(),v.fn.toLocaleString()]){const cell=document.createElement('td');cell.textContent=value;row.append(cell)}body.append(row)}table.append(body);wrap.append(table);el.append(wrap);$('samples').append(el)}if(!filtered.length)$('samples').textContent='没有符合筛选条件的样本。'}
for(const id of ['difficulty','role','search','seed'])$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render()});$('prev').onclick=()=>{page--;render();$('gallery').scrollIntoView()};$('next').onclick=()=>{page++;render();$('gallery').scrollIntoView()};$('closeZoom').onclick=()=>$('zoom').close();render();
</script></body></html>'''
    evidence = dict(environment=env, base_reuse=reuse, thresholds=threshold, integrity=integrity,
                    protocol=read(root / 'protocol.json'))
    content = content.replace('<h2>大棚、建筑与其他困难组</h2>', '<h2>大棚、建筑与其他困难组</h2><p>下表各组包含该标注下的正图和负图：大棚组 13 张（11 正、2 负），建筑组 6 张（4 正、2 负）。组内负图的误检也计入 IoU 分母。</p>')
    content = content.replace('<h3>纯负样本的误检</h3>', '<h3>与此前正样本统计口径对齐</h3><div class="scroll"><table><thead><tr><th>仅含光伏的图</th><th>图数</th><th>A2 IoU</th><th>新完整 0.5</th><th>新完整校准</th><th>旧完整校准</th></tr></thead><tbody>@@POSITIVE_GROUPS@@</tbody></table></div><p>这两行用于对应此前“大棚 11 张、复杂建筑 4 张”的结果；主表同时纳入了同组的负图。纯负图误检率降低也可能伴随前景召回下降，需结合召回率和逐图结果判断。</p><h3>纯负样本的误检</h3>')
    content = content.replace('<h2>如何判断这一步是否成功</h2>', '<h2>当前瓶颈与下一步</h2><p>@@INTERPRETATION@@</p>')
    replacements = {'RESULT': result, 'COMPARISON': comparison, 'FINAL': pct(final), 'CAL': pct(calibrated),
        'A2': pct(a2), 'BASE': pct(base), 'TABLE': ''.join(table_rows), 'GROUPS': ''.join(group_rows),
        'NEG': ''.join(neg_rows), 'SUPPORT': ''.join(support_rows), 'SELECTION': selection_rows,
        'STEP': str(summary['selected_step']), 'CHART': data_uri(chart(root, history, curve)),
        'OPTIONS': ''.join('<option value="' + e(v, quote=True) + '">' + e(v) + '</option>' for v in difficulties),
        'DATA': payload, 'EVIDENCE': e(json.dumps(evidence, ensure_ascii=False, indent=2)), 'COMMIT': env['source_commit']}
    replacements.update(INTERPRETATION=interpretation, POSITIVE_GROUPS=''.join(positive_group_rows))
    for key, value in replacements.items():
        content = content.replace('@@' + key + '@@', value)
    assert '@@' not in content
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding='utf-8')
    print(json.dumps(dict(output=str(output), bytes=output.stat().st_size, final_fixed=final,
        final_calibrated=calibrated, A2=a2, frozen_base=base, old_final_calibrated=old_final,
        integrity=integrity['status']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--old', type=Path, default=REPO / 'docs/experiments/a3_learned_20261029')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    build(args.input, args.old, args.output)
