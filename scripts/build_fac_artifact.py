"""Assemble the FAC 2026 artifact from formal local evidence only."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'artifacts/diverseguide-plus-artifact'
REPORT=Path('/home/lxk/reports/diverse-guide-plus-report')
MAIN=ROOT/'results/fac-writing-materials-20261001-v2'
SUP=ROOT/'results/repeat30-and-json-entry-grid-20261002-v2'
SERVER=ROOT/'results/server-audit-20260930/groups1-4-latest-results-20260930-v1'
ABL=Path('/tmp/results/group2-aggregation-ablation-20261005-v1')
EOS=ROOT/'.worktrees/eos-reentry'
EXCLUDE=shutil.ignore_patterns('.git','.venv','__pycache__','target','build','*.so','*.dll','*.pyc','*.egg-info','.pytest_cache')

def read(p):return json.loads(p.read_text())
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
def csvwrite(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def code():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('src','regex_dfa_guide'):
        shutil.copytree(EOS/name,OUT/name,ignore=EXCLUDE,dirs_exist_ok=True)
    for name in ('metrics.py','string_kernel.py','string_kernel_py.py'):
        copy(ROOT/'src/diverse_guide/evaluation'/name,OUT/'src/diverse_guide/evaluation'/name)
    shutil.copytree(ROOT/'native/wd_kernel',OUT/'native/wd_kernel',ignore=EXCLUDE,dirs_exist_ok=True)
    copy(ROOT/'scripts/build_wd_kernel.py',OUT/'scripts/build_wd_kernel.py')
    copy(ROOT/'LICENSE',OUT/'LICENSE')
    cff=(ROOT/'CITATION.cff').read_text()
    cff=cff[:cff.index('preferred-citation:')]
    cff=cff.replace('please cite the associated ICFEM paper.','please cite the associated FAC manuscript.').replace('version: 0.2.0','version: fac-2026-20261006').replace('title: "DiverseGuide: Automata-Based Steering for Diverse Structured Generation"','title: "DiverseGuide+: FAC 2026 artifact"')
    cff+='date-released: "2026-10-06"\npreferred-citation:\n  type: manuscript\n  title: "Towards Practical Automata-Based Steering of Large Language Models for Diverse Structured Generation"\n  authors:\n'
    for last,first in [('Luan','Xiaokun'),('Wei','Zeming'),('Zhang','Yihao'),('Wang','Kailong'),('Sun','Meng')]:cff+=f'    - family-names: {last}\n      given-names: {first}\n'
    cff+='  year: 2026\n  notes: "FAC revision manuscript; publication DOI is not yet assigned."\n'
    (OUT/'CITATION.cff').write_text(cff)
    # Keep the existing native/package layout so uv and maturin work unchanged.
    project=(EOS/'pyproject.toml').read_text().replace('version = "0.2.0"','version = "0.3.0"',1)
    project=project.replace('"transformers>=5.3.0"','"transformers==5.3.0"').replace('"numpy>=2.2.3"','"numpy==2.2.3"').replace('"scipy>=1.15.2"','"scipy==1.15.2"')
    project=project.replace('[dependency-groups]','[dependency-groups]\nwriting = ["matplotlib>=3.10", "numpy==2.2.3", "scipy==1.15.2"]\nlibrary = ["coverage==7.10.7", "pytest>=9.0.2", "pytest-cov>=7.1.0", "dnspython==2.8.0", "idna==3.10"]')
    (OUT/'pyproject.toml').write_text(project)
    env=OUT/'environment';env.mkdir(exist_ok=True)
    save(env/'generation-versions.json',read(MAIN/'methods/server-generation-environment.json'))
    configs=read(MAIN/'methods/main-configurations.json')
    models={r['model']:r['model_revision'] for r in configs}
    save(env/'models.json',[dict(id=m,revision=r) for m,r in models.items()])
    grammars={r['grammar']:{k:r[k] for k in ('grammar','regex','prompt','max_tokens')} for r in configs}
    for name,g in grammars.items():save(OUT/'configs/grammars'/f'{name}.json',g)
    for name,c in dict(main=dict(models=list(models),grammars=list(grammars),methods=['outlines','dg','dgplus'],seeds=list(range(42,47))),temperature=dict(models=['Qwen/Qwen2.5-1.5B-Instruct'],grammars=list(grammars),methods=['outlines','dgplus'],temperatures=[.7,1,1.3,1.5,2],seeds=list(range(42,47))),sensitivity=dict(models=['Qwen/Qwen2.5-1.5B-Instruct'],grammars=list(grammars),methods=['dgplus'],gammas=[.1,.3,.5],etas=[0,.5,1,2],seeds=list(range(42,47))),ablation=dict(models=['Qwen/Qwen2.5-1.5B-Instruct'],grammars=list(grammars),methods=['dgplus'],gammas=[.1,.3,.5],etas=[0],estimators=['min','harmonic'],seeds=list(range(42,47))),repeat30=dict(models=['Qwen/Qwen2.5-1.5B-Instruct'],grammars=['json','email','css-color'],methods=['dg','dgplus'],seeds=list(range(100000,100030)))).items():
        c.update(valid_samples=500,temperature=c.get('temperature',1),gamma=.5,eta=.5,top_k=0,top_p=1,repetition_penalty=1,batch_size=1,dtype='float16')
        save(OUT/'configs/experiments'/f'{name}.json',c)
    save(OUT/'configs/experiments/library.json',dict(grammars=['email','css-color','semver'],budgets=read(MAIN/'figure1/manifest.json').get('budgets'),seeds=list(range(42,47)),models=list(models),methods=['outlines','dg','dgplus']))
    # Internal selection occurs after DFA compilation; callers do not expose it.
    p=OUT/'src/diverse_guide/guide_rust.py';s=p.read_text()
    json_hash=hashlib.sha256(grammars['json']['regex'].encode()).hexdigest()
    s=s.replace('import torch\n','import hashlib\n\nimport torch\n',1)
    s=s.replace('eos_completion_mode: str = "legacy"','eos_completion_mode: str = "auto"')
    needle='        self.dfa = guide.dfa\n'
    s=s.replace(needle,needle+f'        if eos_completion_mode == "auto":\n            constraint_key = hashlib.sha256(regex_string.encode("utf-8")).hexdigest()\n            eos_completion_mode = "entry_count" if constraint_key == "{json_hash}" else "legacy"\n',1)
    s=s.replace('        self.eos_completion_mode = eos_completion_mode\n\n\ndef diverse_regex','        self.eos_completion_mode = logits_processor.eos_completion_mode\n\n\ndef diverse_regex')
    p.write_text(s)
    # Historical comparison implementations are necessary to rerun engineering measurements.
    for method in ('old','plus'):
        source=ROOT/'results/table8-remeasure-20261002/sources'/method
        shutil.copytree(source,OUT/'src/engineering_versions'/method,ignore=EXCLUDE,dirs_exist_ok=True)
    frozen=MAIN/'figure1/frozen'
    for name in ('email_validator','webcolors','semver','tests'):
        shutil.copytree(frozen/name,OUT/'library'/name,ignore=EXCLUDE,dirs_exist_ok=True)
    copy(frozen/'pyproject.toml',OUT/'library/pyproject.toml')
    copy(frozen/'.coveragerc',OUT/'library/.coveragerc')
    # Independent figure entry points, with shared styling and portable paths.
    src=(REPORT/'figures/src/make_revision_figures.py').read_text()
    start=src.index('ROOT = ');end=src.index('\nGRAMMARS =',start)
    src=src[:start]+'''ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/datasets"
SUPPLEMENT = ROOT / "results/datasets"
OUT_FIG = ROOT / "results/rendered/figures"
OUT_DATA = OUT_FIG / "data"
LIBRARY_SUMMARY = ROOT / "results/datasets/figure1/summary.csv"
'''+src[end:]
    src=src.replace('"text.usetex": True','"text.usetex": False')
    src=src.replace('r"\\textsc{DiverseGuide}"','"DG"').replace('r"\\textsc{DiverseGuide}\\textsuperscript{+}"','"DG+"')
    src=src.replace('r"\\texttt{email\\_validator}"','"email_validator"').replace('r"\\texttt{webcolors}"','"webcolors"').replace('r"\\texttt{python-semver}"','"python-semver"').replace('\\\\%','%')
    src=src[:src.index('\ndef main():')]
    (OUT/'scripts/figure_common.py').write_text(src)
    for name,load,plot in [('temperature','load_temperature','plot_temperature'),('sensitivity','load_sensitivity','plot_sensitivity'),('library','load_library','plot_library')]:
        (OUT/f'scripts/figure_{name}.py').write_text(f'''"""Render the paper {name} figure from saved five-seed summaries."""
import figure_common as common
common.OUT_FIG.mkdir(parents=True, exist_ok=True)
common.OUT_DATA.mkdir(parents=True, exist_ok=True)
rows = common.{load}()
common.write_rows(rows, common.OUT_DATA / "{name}.csv")
common.{plot}(rows)
''')
    # Preserve numerical table rendering logic; separate entry points use only their data.
    src=(REPORT/'figures/src/make_revision_tables.py').read_text()
    start=src.index('ROOT =');end=src.index('\nMODELS =',start)
    src=src[:start]+'ROOT = Path(__file__).resolve().parents[1]\nPACKAGE = ROOT / "results/datasets"\nSUPPLEMENT = PACKAGE\n'+src[end:]
    src=src.replace('PACKAGE / "data" /','PACKAGE /').replace('ROOT / "local-archives" / "group2-aggregation-ablation-20261005-v1"','PACKAGE / "ablation"')
    if '\nif __name__' in src:src=src[:src.index('\nif __name__')]
    (OUT/'scripts/table_common.py').write_text(src)
    save(env/'versions.json',dict(version='fac-2026-20261006',generation_source_commit='67ead252ad20529d29111f07687b99fd06870ba2',native_kernel_source_sha256=digest(ROOT/'native/wd_kernel/wd_kernel.c'),outlines='1.2.12',engineering_comparison_commits=read(ROOT/'results/table8-remeasure-20261002/protocol.json')['commits'],paper_title='Towards Practical Automata-Based Steering of Large Language Models for Diverse Structured Generation'))


def evidence():
    datasets=OUT/'results/datasets'
    for g in ('group1','group2','group3'):
        for p in (MAIN/'data'/g).glob('*'):
            if p.is_file() and p.suffix in ('.csv','.json') and 'provenance' not in p.name:copy(p,datasets/g/p.name)
    for g in ('repeat30','json-entry-grid'):
        for p in (SUP/g/'analysis').glob('*'):
            if p.suffix in ('.json','.csv') and p.stem!='provenance':copy(p,datasets/g/'analysis'/p.name)
    for name in ('summary.csv','endpoint-table.csv','per-run.csv','manifest.json'):
        if (MAIN/'figure1'/name).exists():copy(MAIN/'figure1'/name,datasets/'figure1'/name)
    for p in (ABL/'analysis').glob('*'):
        if p.suffix in ('.csv','.json'):copy(p,datasets/'ablation/analysis'/p.name)
    for p in (MAIN/'statistics').glob('*.csv'):
        if p.name.startswith('candidate-cell') or p.name=='input-main-per-run.csv':copy(p,datasets/'statistics'/p.name)
    for name in ('memory-MiB.csv',):copy(MAIN/'tables'/name,datasets/'tables'/name)
    for name in ('structure_sizes.csv','memory_runs.json','memory_summary.csv'):copy(MAIN/'data/group4'/name,datasets/'group4'/name)
    catalog=[]
    def add(group,row,source,evaluation,configuration):
        payload=read(source)
        model=payload['model'];grammar=payload['grammar'];seed=payload['seed']
        method=row.get('method') or 'diverse_guide_plus'
        suffix=row.get('view_id',row.get('source_task_id',source.parent.name)).replace('__entry_count__','__')
        # A view is a formal experiment setting; shared generation inputs may occur across groups.
        base=Path(group)/model.split('/')[-1]/grammar/method/suffix
        samples=OUT/'results/samples'/Path(str(base)+'.jsonl')
        samples.parent.mkdir(parents=True,exist_ok=True)
        samples.write_text(''.join(json.dumps(dict(text=s),ensure_ascii=False)+'\n' for s in payload['samples']))
        record=read(evaluation)
        save(OUT/'results/metrics'/Path(str(base)+'.json'),record)
        cfg=read(configuration)
        save(OUT/'results/configurations'/Path(str(base)+'.json'),cfg)
        item=dict(group=group,model=model,grammar=grammar,method=method,seed=seed,gamma=cfg['gamma'],eta=cfg['completion_bias_eta'],estimator=cfg['path_frequency_estimator'],temperature=cfg['generation_config']['temperature'],eos_completion_mode=cfg.get('eos_completion_mode','legacy'),valid_samples=len(payload['samples']),source_result_sha256=digest(source),samples_path=str(Path('results/samples')/Path(str(base)+'.jsonl')),metrics_path=str(Path('results/metrics')/Path(str(base)+'.json')),configuration_path=str(Path('results/configurations')/Path(str(base)+'.json')))
        status=source.parent/'status.json'
        if status.exists():
            state=read(status)
            for key in ('accepted','rejected','generated_tokens','total_generated_tokens','token_cost'): 
                if key in state:item[key]=state[key]
        catalog.append(item)
    for r in read(MAIN/'raw/RUN_CATALOG.json'):
        if r['role']!='formal' or r['group'] not in ('group1','group2','group3'):continue
        if r['group']=='group2' and r['grammar']=='json':continue # final sensitivity uses updated JSON grid
        ev=MAIN/'reevaluation/evaluations'/r['package_evaluation']
        assert ev.exists(),ev
        add({'group1':'main','group2':'sensitivity','group3':'temperature'}[r['group']],r,SERVER/r['package_result'],ev,SERVER/r['package_configuration'])
    for r in read(SUP/'RUN_CATALOG.json'):
        if r['role']!='formal':continue
        add('repeat30' if r['group']=='repeat30' else 'sensitivity',r,SUP/r['package_result'],SUP/r['package_evaluation'],SUP/r['package_configuration'])
    # All 180 formal ablation cells, including verified reused JSON cells.
    manifest=read(ABL/'manifest.json')
    for task in manifest['tasks']:
        if task.get('stage') not in ('production','formal'):continue
        taskdir=ABL/'production/tasks'/task['id']
        state=read(taskdir/'status.json')
        def relocate(value):
            if ABL.name+'/' in value:
                return ABL/value.split(ABL.name+'/',1)[1]
            raise ValueError(value)
        source=relocate(state['result_path'])
        evaluation=relocate(state['evaluation_path'])
        configuration=evaluation.parent/'configuration.json'
        assert source.is_file() and evaluation.is_file() and configuration.is_file()
        add('ablation',dict(method='diverse_guide_plus',source_task_id=task['id']),source,evaluation,configuration)
    for imported in (ABL/'imports').glob('production__json*'):
        ev=list((imported/'attempts').rglob('evaluation.json'))
        assert len(ev)==1,imported
        add('ablation',dict(method='diverse_guide_plus',source_task_id=imported.name),imported/'result.json',ev[0],ev[0].parent/'configuration.json')
    save(OUT/'results/RUN_CATALOG.json',catalog)
    csvwrite(OUT/'results/RUN_CATALOG.csv',catalog)
    # Compact library reports include counters/denominators at every budget, not duplicated library trees.
    for p in (MAIN/'figure1/curves').rglob('coverage.json'):
        copy(p,OUT/'results/library'/p.relative_to(MAIN/'figure1/curves'))
    # Keep actual engineering raw data, excluding locks, trial runs, binaries and clock pilots.
    for group,source in [('timing',ROOT/'results/table8-remeasure-20261002'),('memory',ROOT/'results/group4-memory-workstation-20260930-v1')]:
        dest=OUT/'results/efficiency'/group
        for name in ('protocol.json','manifest.json','table8.csv','table8.json','memory_runs.json','memory_summary.csv','structure_sizes.csv','environment.json','distributions.json'):
            if (source/name).exists():copy(source/name,dest/name)
        if group=='timing':
            for p in (source/'runs').glob('*.json'):copy(p,dest/'runs'/p.name)
            shutil.copytree(source/'traces',dest/'traces',dirs_exist_ok=True)
        else:
            for p in (source/'production').rglob('measurement.json'):copy(p,dest/'raw'/p.relative_to(source/'production'))
    print('Formal views:',len(catalog),flush=True)


if __name__=='__main__':
    code();evidence()
