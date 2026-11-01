# -*- coding: utf-8 -*-
"""Build a local visual comparison from independently verified CV predictions."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_hard_negative import NAMES, reconstruct, stats


def percent(value):
    return '—' if value is None else f'{value*100:.2f}%'


def error_overlay(rgb,pred,gt):
    out=(rgb.astype(np.float32)*.4).astype(np.uint8)
    out[pred & gt]=(.32*rgb[pred & gt]+.68*np.array([48,193,132])).astype(np.uint8)
    out[pred&~gt]=[245,65,78]
    out[~pred & gt]=[50,155,255]
    return Image.fromarray(out)


def main(args):
    data=args.data
    summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((data/'independent_reconstruction.json').read_text(encoding='utf-8'))
    assert audit['status']=='passed'
    recommended,alternate=audit['recommended'],audit['best_development_iou']
    if alternate==recommended:
        alternate='residual_balanced' if recommended!='residual_balanced' else 'residual_texture'
    output=data/'report'
    (output/'panels').mkdir(parents=True,exist_ok=True)
    root=args.root
    with (root/'manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows={r['sample_id']:r for r in csv.DictReader(stream)}
    annotations={r['sample_id']:r for r in json.loads((REPO/'configs/a0a1_selected_20261029.json').read_text(encoding='utf-8'))['records']}
    score=np.load(REPO/'outputs/greenhouse_pilot_20261030/greenhouse_pilot/score_grids.npz')
    old=np.load(REPO/'outputs/dynamic_brightness_20261030/dynamic_brightness/weight_grids.npz')
    learned=np.load(data/'holdout_grids.npz')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    catalog=[]
    for sid in summary['subset_ids']['all']:
        with Image.open(root/rows[sid]['image']) as im:
            rgb=np.array(im.convert('RGB'))
        with Image.open(root/rows[sid]['label']) as im:
            gt=np.asarray(im)==255
        for seed in (42,43,44):
            prefix=f'{sid}_{seed}_'
            d=score[prefix+'spatial']
            s=np.clip(score[prefix+'pv_semantic']-score[prefix+'semantic_penalty']*old[sid+'_gate'],-1,1)
            masks={}
            for arm in ('local',recommended,alternate):
                a=old[prefix+'local_alpha'] if arm=='local' else learned[prefix+arm+'_alpha']
                r=np.zeros_like(a) if arm=='local' else learned[prefix+arm+'_residual']
                masks[arm]=reconstruct(d,s,a,r)>=summary['records'][arm][sid][str(seed)]['threshold']
            def label(arm):
                row=stats({k:summary['records'][arm][sid][str(seed)][k] for k in ('tp','fp','fn','tn')})
                return f"IoU {percent(row['iou'])}  误检 {row['fp']:,}  召回 {percent(row['recall'])}"
            images=[Image.fromarray(rgb),Image.fromarray(gt.astype(np.uint8)*255),error_overlay(rgb,masks['local'],gt),
                    error_overlay(rgb,masks[recommended],gt),error_overlay(rgb,masks[alternate],gt),
                    Image.fromarray(masks[recommended].astype(np.uint8)*255)]
            titles=['原始 RGB','可信光伏掩膜','原动态 alpha：误差', '推荐候选：误差','比较候选：误差','推荐候选：预测掩膜']
            subtitles=[sid,annotations[sid]['difficulty'],label('local'),label(recommended),label(alternate),'绿色正确光伏 · 红色误检 · 蓝色漏检']
            panel=Image.new('RGB',(1032,804),'#f3f6f8')
            draw=ImageDraw.Draw(panel)
            draw.text((16,9),f'{sid} · 支持选择 {seed} · 空间组留出预测',fill='#142a3a',font=font)
            for i,im in enumerate(images):
                x=12+(i%3)*340;y=40+(i//3)*376
                draw.text((x,y),titles[i],fill='#142a3a',font=font)
                draw.text((x,y+24),subtitles[i],fill='#526676',font=small)
                thumb=im.resize((328,328),resample=Image.Resampling.NEAREST)
                panel.paste(thumb,(x,y+48))
            name=f'{sid}_{seed}.png'
            panel.save(output/'panels'/name)
            catalog.append(dict(id=sid,seed=seed,scene=rows[sid]['source_scene'],group=annotations[sid]['difficulty'],
                                file='panels/'+name))
    rows_html=[]
    for arm in summary['records']:
        g=audit['groups']['greenhouse'][arm];a=audit['groups']['all'][arm];n=audit['groups']['pure_negative'][arm]
        tag='推荐' if arm==recommended else '备选' if arm==alternate else ''
        rows_html.append(f'<tr><td>{NAMES[arm]} <b>{tag}</b></td><td>{percent(a["iou"])}</td><td>{percent(g["iou"])}</td><td>{percent(g["recall"])}</td><td>{g["fp"]:,.0f}</td><td>{n["fp"]:,.0f}</td></tr>')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>光伏背景误检迭代 · 空间组留出</title><style>
body{font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#f2f6f8;color:#183449}main{max-width:1120px;margin:auto;padding:28px 20px 70px}h1{font-size:29px;margin-bottom:8px}h2{font-size:21px}.card{background:white;border:1px solid #dbe4e9;border-radius:12px;padding:18px;margin:20px 0}.hint{color:#526676}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #e2e8eb}th{background:#edf5f6}b{color:#00745d}select{font:inherit;padding:7px;margin-right:12px;max-width:100%}img{width:100%;height:auto;display:block}a{color:#0c697c}label{display:inline-block;margin-bottom:8px}button{padding:9px 15px;border:1px solid #c8d7dc;border-radius:7px;background:#edf5f6;cursor:pointer}#counter{margin-left:12px}.red{color:#da243e}.blue{color:#0876cc}</style>
<main><h1>光伏背景误检迭代</h1><p>24 张开发图 · 5 个地理空间组交叉验证 · 3 次独立支持选择</p>
<div class="card"><p><b>推荐：@@RECOMMENDED@@</b>；比较候选：@@ALTERNATE@@。</p><p>当前展示均为排除所属完整空间组后的预测。可信光伏掩膜之外全部作为背景监督。模型现在使用其他开发图的标注训练，已有额外监督；这些结果属于开发交叉验证，不能当作新区域独立测试或完整 SSPNet 成绩。</p><p class="hint">早期 V1/V2 只按来源文件留出，部分来源存在地理关联。此页使用 V3 修正后的 5 组结果。</p></div>
<div class="card table"><table><thead><tr><th>方案</th><th>全24图 IoU</th><th>大棚13图 IoU</th><th>大棚光伏召回</th><th>大棚背景 FP</th><th>纯负7图 FP</th></tr></thead><tbody>@@ROWS@@</tbody></table><p class="hint">每次支持先汇总计数，再平均三次指标。FP 为每次支持的误报像素均值，不是大棚类别的独立像素标注。</p></div>
<div class="card"><h2>逐图检查</h2><p>推荐：@@RECOMMENDED@@；比较：@@ALTERNATE@@。<span class="red">红色＝误检</span>，<span class="blue">蓝色＝漏检</span>，绿色＝正确光伏。</p>
<label>图像 <select id="sample"></select></label><label>支持选择 <select id="seed"><option>42</option><option>43</option><option>44</option></select></label><button id="prev">上一张</button> <button id="next">下一张</button><span id="counter"></span><p id="detail" class="hint"></p><a id="full" target="_blank"><img id="panel" alt="原图、光伏掩膜与三种模型的预测误差"></a></div>
<div class="card"><h2>验证与使用范围</h2><p>已独立重建 @@COUNT@@ 组逐图混淆计数；核对 @@HASHES@@ 个 checkpoint 哈希。候选模型另使用全部24图训练并保存，需在新空间区域验证；此页展示的是各折模型的预测，不能声称最终单一模型已取得同样成绩。</p><p>冻结基础编码器，输出仍是 CAM 先验；尚未接入 SSPNet 的 DCMA 和解码器。频谱描述为常规局部 FFT 能量特征，是否有增量收益请看相同容量的“FFT通道置零”对照。</p></div></main>
<script>const items=@@CATALOG@@;const ids=[...new Set(items.map(x=>x.id))];const sample=document.getElementById('sample'),seed=document.getElementById('seed');for(const id of ids){const x=items.find(x=>x.id===id);sample.add(new Option(id+' · '+x.group,id));}function update(){const x=items.find(x=>x.id===sample.value&&String(x.seed)===seed.value);document.getElementById('panel').src=x.file;document.getElementById('full').href=x.file;document.getElementById('detail').textContent=x.scene+' · '+x.group;document.getElementById('counter').textContent=(ids.indexOf(x.id)+1)+' / '+ids.length;}sample.onchange=seed.onchange=update;document.getElementById('prev').onclick=()=>{sample.selectedIndex=(sample.selectedIndex+ids.length-1)%ids.length;update()};document.getElementById('next').onclick=()=>{sample.selectedIndex=(sample.selectedIndex+1)%ids.length;update()};update();</script></html>'''
    for key,value in {'RECOMMENDED':NAMES[recommended],'ALTERNATE':NAMES[alternate], 'ROWS':''.join(rows_html),
                      'CATALOG':json.dumps(catalog,ensure_ascii=False),'COUNT':str(audit['count_comparisons']),
                      'HASHES':str(audit['checkpoint_hashes'])}.items():
        text=text.replace('@@'+key+'@@',value)
    (output/'index.html').write_text(text,encoding='utf-8')
    for path in (output/'panels').glob('*.png'):
        with Image.open(path) as im:
            im.verify()
    (output/'report_qa.json').write_text(json.dumps(dict(panels=len(catalog),all_images_decode=True,
        recommended=recommended,alternate=alternate),indent=2),encoding='utf-8')
    print(output/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(r'G:\PV-GF\PV-896'))
    args=p.parse_args();torch.set_num_threads(2);main(args)
