"""Apply a saved-sample Vendi reassessment to a copied supplement package."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from scipy.stats import t


def read(p):
    return json.loads(p.read_text())


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def describe(values):
    n = len(values)
    mean, std = statistics.mean(values), statistics.stdev(values)
    margin = float(t.ppf(.975, n - 1)) * std / math.sqrt(n)
    return dict(n_runs=n, mean=mean, std=std, ci95_low=mean-margin, ci95_high=mean+margin, ci95_df=n-1)


def table(p, rows):
    save(p.with_suffix('.json'), rows)
    with p.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--recheck', type=Path, required=True)
    args = parser.parse_args()
    root, recheck = args.package.resolve(), args.recheck.resolve()
    ledger = read(recheck/'ledger.json')
    assert len(ledger) == 321
    by_sha = {r['result_sha256']: r for r in ledger}
    for row in ledger:
        assert digest(root/row['package_result']) == row['result_sha256']
        shutil.copy2(recheck/'evaluations'/row['package_evaluation'], root/row['package_evaluation'])
    audit = root/'provenance'/'vendi-reevaluation-20261002'
    audit.mkdir(parents=True, exist_ok=True)
    for name in ('ledger.json', 'verification.json'):
        shutil.copy2(recheck/name, audit/name)
    shutil.copytree(recheck/'unique', audit/'unique', dirs_exist_ok=True)
    analyses = {}
    for group in ('repeat30', 'json-entry-grid'):
        folder = root/group/'analysis'
        files = {p.stem: read(p) for p in folder.glob('*.json') if p.stem != 'provenance'}
        for name in ('per-run', 'legacy-per-run'):
            for row in files.get(name, []):
                item = by_sha[row['result_sha256']]
                row['vendi'] = item['new_vendi']
                row['evaluation_sha256'] = item['new_evaluation_sha256']
        if group == 'repeat30':
            index = {(r['grammar'], r['method'], r['seed']): r for r in files['per-run']}
            for row in files['summary-long']:
                if row['metric'] == 'vendi':
                    row.update(describe([index[row['grammar'], row['method'], s]['vendi'] for s in range(100000,100030)]))
            for row in files['paired-per-seed']:
                if row['metric'] == 'vendi':
                    row['difference'] = index[row['grammar'], 'diverse_guide_plus', row['seed']]['vendi'] - index[row['grammar'], 'diverse_guide', row['seed']]['vendi']
            for row in files['paired-summary']:
                if row['metric'] == 'vendi':
                    values = [p['difference'] for p in files['paired-per-seed'] if p['grammar']==row['grammar'] and p['metric']=='vendi']
                    row.update(describe(values))
                    row.update(higher=sum(v>1e-10 for v in values), tied=sum(abs(v)<=1e-10 for v in values), lower=sum(v<-1e-10 for v in values))
        else:
            index = {(r['gamma'], r['eta'], r['seed'], r['mode']): r for r in files['per-run']+files['legacy-per-run']}
            for row in files['summary-long']:
                if row['metric']=='vendi':
                    row.update(describe([index[row['gamma'],row['eta'],s,row['mode']]['vendi'] for s in range(42,47)]))
            for row in files['paired-entry-vs-legacy-per-seed']:
                if row['metric']=='vendi':
                    g,e,s=row['gamma'],row['eta'],row['seed']
                    row['difference']=index[g,e,s,'entry_count']['vendi']-index[g,e,s,'legacy']['vendi']
            for row in files['paired-entry-vs-legacy-summary']:
                if row['metric']=='vendi':
                    row.update(describe([p['difference'] for p in files['paired-entry-vs-legacy-per-seed'] if p['gamma']==row['gamma'] and p['eta']==row['eta'] and p['metric']=='vendi']))
            for row in files['paired-grid-vs-g05-e05']:
                if row['metric']=='vendi':
                    row.update(describe([index[row['gamma'],row['eta'],s,'entry_count']['vendi']-index[.5,.5,s,'entry_count']['vendi'] for s in range(42,47)]))
        for name, rows in files.items():
            old = read(root/'provenance'/'original-derived'/group/'analysis'/f'{name}.json')
            for before, after in zip(old, rows, strict=True):
                if name.endswith('per-run') or name=='per-run':
                    assert {k:v for k,v in before.items() if k not in ('vendi','evaluation_sha256')} == {k:v for k,v in after.items() if k not in ('vendi','evaluation_sha256')}
                elif before.get('metric') != 'vendi':
                    assert before == after
            table(folder/name, rows)
        provenance = read(folder/'provenance.json')
        provenance['vendi_reevaluation'] = dict(date='2026-10-02', version=read(recheck/'verification.json')['version'], ledger='provenance/vendi-reevaluation-20261002/ledger.json', historical_input_hashes='Original input_sha256 fields describe the original server evaluation; current evaluation hashes are in per-run and the reassessment ledger.')
        save(folder/'provenance.json', provenance)
        analyses[group]=files
    report = ['补充实验分析（2026-10-02，Vendi 统一重评）', '', '240 个新增正式 run，120,000 条有效样本；60 个历史 legacy 对照；21 个 smoke。所有样本、seed、生成配置和尝试记录保留，没有重新生成。', '统计单位是 run；均值、样本标准差和逐指标 95% t 区间。30-seed 配对使用 100000–100029，JSON 网格配对使用 42–46。区间未作多重比较校正，未新增 p-value 或显著性结论。', 'Vendi 使用 UTF-8 字节 WD 核（degree=5、shift=1），特征值按 trace 归一化，无对角归一化、无 epsilon；仅裁剪舍入级负特征值。', '', 'A. DG+ − DG：30-seed 配对差值', 'grammar | metric | mean difference | sample SD | 95% pointwise CI | higher/tied/lower']
    for r in analyses['repeat30']['paired-summary']:
        scale=100 if r['metric']=='reject_ratio' else 1
        report.append(f"{r['grammar']} | {r['metric']} | {r['mean']*scale:.4f} | {r['std']*scale:.4f} | [{r['ci95_low']*scale:.4f}, {r['ci95_high']*scale:.4f}] | {r['higher']}/{r['tied']}/{r['lower']}")
    report += ['', '覆盖率及本节拒绝率为百分点；其他指标为原单位。', '', 'B. JSON 网格：五 seed 均值 ± 样本标准差', 'gamma | eta | mode | State % | Transition % | Path % | Vendi | Distinct2 | Distinct3 | Mean chars | Reject % | Tokens/valid']
    lookup={(r['gamma'],r['eta'],r['mode'],r['metric']):r for r in analyses['json-entry-grid']['summary-long']}
    metrics=['state_coverage_pct','transition_coverage_pct','path_coverage_pct','vendi','distinct_2','distinct_3','average_length','reject_ratio','tokens_per_valid']
    for g in (.1,.3,.5):
        for e in (0,.5,1,2):
            for mode in ('entry_count','legacy'):
                cells=[]
                for metric in metrics:
                    r=lookup[g,e,mode,metric]; scale=100 if metric=='reject_ratio' else 1
                    cells.append(f"{r['mean']*scale:.4f} ± {r['std']*scale:.4f}")
                report.append(f'{g:g} | {e:g} | {mode} | '+' | '.join(cells))
    report += ['', 'C. 同 gamma、eta、seed：entry_count − legacy', 'gamma | eta | metric | mean difference | sample SD | 95% pointwise CI']
    for r in analyses['json-entry-grid']['paired-entry-vs-legacy-summary']:
        scale=100 if r['metric']=='reject_ratio' else 1
        report.append(f"{r['gamma']:g} | {r['eta']:g} | {r['metric']} | {r['mean']*scale:.4f} | {r['std']*scale:.4f} | [{r['ci95_low']*scale:.4f}, {r['ci95_high']*scale:.4f}]")
    report += ['', '网格内相对 entry_count γ=.5、η=.5 的配对差值见 json-entry-grid/analysis/paired-grid-vs-g05-e05.csv。', '写作时应同时查看 coverage、多样性与拒绝率/token 成本，不能仅凭 Vendi 选择共同配置。重评前表格仅保存在 provenance/original-derived/，不得用于当前数值引用。']
    (root/'ANALYSIS.txt').write_text('\n'.join(report)+'\n')
    manifest=read(root/'PACKAGE_MANIFEST.json')
    manifest.update(name=root.name, created_at='2026-10-02', original_package='repeat30-and-json-entry-grid-20261001-v1', vendi_reevaluation=read(recheck/'verification.json'))
    save(root/'PACKAGE_MANIFEST.json',manifest)
    counts={}
    for r in ledger:
        counts.setdefault(r['group'], []).append(abs(r['difference']))
    changes=dict(max_absolute_score_change=max(abs(r['difference']) for r in ledger), views_changed_over_1e_6=sum(abs(r['difference'])>1e-6 for r in ledger), per_group_max_absolute_change={k:max(v) for k,v in counts.items()})
    save(audit/'change-summary.json',changes)
    print(json.dumps(changes, indent=2))


if __name__=='__main__':
    main()
