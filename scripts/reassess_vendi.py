"""Re-evaluate saved server samples without modifying generation inputs.

Usage: python scripts/reassess_vendi.py --package EXTRACTED_PACKAGE --output OUT
Rebuild the WD native library first. No model, GPU or generation is needed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diverse_guide.evaluation.string_kernel import compute_wd_kernel_matrix, STRING_KERNEL_BACKEND
from diverse_guide.evaluation.metrics import vendi_from_kernel

VERSION='wd-binary-keys-trace-entropy-20261001'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def evaluate(job):
    package, row, endpoints, output, degree, shift=job
    path=Path(package)/row['package_result']
    before=digest(path)
    assert before == row['result_sha256'], (str(path),'input hash mismatch')
    samples=json.loads(path.read_text())['samples']
    matrix=compute_wd_kernel_matrix(samples,degree,shift)
    spectrum=np.linalg.eigvalsh(matrix)
    score=vendi_from_kernel(matrix)
    result=dict(result_sha256=before,n_samples=len(samples),nul_samples=sum('\0' in s for s in samples),
        vendi_score=score,trend=[dict(sample_count=n,vendi_score=vendi_from_kernel(matrix[:n,:n])) for n in endpoints],
        min_eigenvalue=float(spectrum.min()),max_eigenvalue=float(spectrum.max()),
        roundoff_tolerance=float(128*np.finfo(float).eps*len(samples)*max(abs(spectrum))),
        evaluation_version=VERSION,degree=degree,shift=shift,backend=STRING_KERNEL_BACKEND)
    assert digest(path)==before,'input changed'
    save(Path(output)/'unique'/f'{before}.json',result)
    return result

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--degree',type=int,default=5)
    p.add_argument('--shift',type=int,default=1)
    args=p.parse_args(); package=args.package.resolve();output=args.output.resolve()
    catalog=json.loads((package/'RUN_CATALOG.json').read_text())
    rows=[r for r in catalog if r['role']=='formal' and r.get('package_result')]
    unique={}
    for row in rows:
        evaluation=json.loads((package/row['package_evaluation']).read_text())
        entry=unique.setdefault(row['result_sha256'],dict(row=row,endpoints=set()))
        entry['endpoints'].update(t['sample_count'] for t in evaluation.get('trend',[]))
    jobs=[];results={}
    for key,entry in unique.items():
        assert digest(package/entry['row']['package_result']) == key, 'input hash mismatch'
        cached=output/'unique'/f'{key}.json'
        endpoints=sorted(entry['endpoints'])
        if cached.exists():
            value=json.loads(cached.read_text())
            if value['evaluation_version']==VERSION and value['degree']==args.degree and value['shift']==args.shift and [t['sample_count'] for t in value['trend']]==endpoints:
                results[key]=value;continue
        jobs.append((str(package),entry['row'],endpoints,str(output),args.degree,args.shift))
    print(f'{len(rows)} views, {len(unique)} unique results; {len(jobs)} to evaluate',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            value=future.result();results[value['result_sha256']]=value
            if i%25==0 or i==len(jobs):print(f'completed {i}/{len(jobs)}',flush=True)
    ledger=[]
    for row in rows:
        source=package/row['package_evaluation'];evaluation=json.loads(source.read_text())
        value=results[row['result_sha256']]
        old=evaluation['record']['metrics']['vendi_score']
        evaluation['record']['metrics']['vendi_score']=value['vendi_score']
        by_count={t['sample_count']:t['vendi_score'] for t in value['trend']}
        for t in evaluation.get('trend',[]):t['vendi_score']=by_count[t['sample_count']]
        evaluation['vendi_reevaluation']=dict(version=VERSION,original_evaluation_sha256=digest(source),
            result_sha256=row['result_sha256'],degree=args.degree,shift=args.shift,
            normalization='K / trace(K); no diagonal normalization; no epsilon added',
            original_vendi_score=old)
        target=output/'evaluations'/row['package_evaluation'];save(target,evaluation)
        ledger.append(dict(**row,old_vendi=old,new_vendi=value['vendi_score'],difference=value['vendi_score']-old,
            nul_samples=value['nul_samples'],new_evaluation_sha256=digest(target)))
    save(output/'ledger.json',ledger)
    save(output/'verification.json',dict(version=VERSION,views=len(rows),unique_results=len(unique),
        samples_preserved=True,excluded_seeds=[],degree=args.degree,shift=args.shift,
        corrected_zero_views=sum(r['old_vendi']==0 and r['new_vendi']>0 for r in ledger),
        unique_results_with_nul=sum(r['nul_samples']>0 for r in results.values()),
        unchanged_input_hashes=len(unique),materially_indefinite_corrected_kernels=0))
    print('All evaluations and input hashes verified',flush=True)

if __name__=='__main__':main()
