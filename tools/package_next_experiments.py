"""Package completed new experiments; no training or test-data analysis."""
from pathlib import Path
import hashlib,json,shutil,sqlite3,zipfile
import pandas as pd
import numpy as np

REPO=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    direct=REPO/'airfare_round2';business=REPO/'airfare_business'
    submissions=direct/'submissions';audit=read(submissions/'inference_audit.json')
    freeze=read(submissions/'frozen_manifest.json');assert audit['manifest_sha256']==sha(submissions/'frozen_manifest.json')
    configs=freeze['snapshot']['models']
    assert set(configs)=={'xgb','lgb','cat64_direct','cat_onehot64','business_specialist'}
    for name,digest in freeze['snapshot']['source_hashes'].items():assert sha(REPO/name)==digest,name
    for name,digest in freeze['snapshot']['business_metadata']['source_hashes'].items():assert sha(REPO/name)==digest,name
    baseline_ids=pd.read_csv(REPO/'submissions/submission_equal_blend_onehot64.csv',usecols=['id'],dtype={'id':str}).id
    for filename,digest in audit['submission_sha256'].items():
        path=submissions/filename;assert sha(path)==digest
        frame=pd.read_csv(path,dtype={'id':str})
        assert list(frame)==['id','price'] and len(frame)==3898 and frame.id.equals(baseline_ids)
        assert np.isfinite(frame.price).all() and (frame.price>=0).all()
        target=REPO/'submissions'/filename
        assert not target.exists() or target.read_bytes()==path.read_bytes()
        shutil.copy2(path,target)
    configuration=dict(models=configs,policy=freeze['snapshot']['policy'],versions=freeze['snapshot']['versions'],
        note='Direct CatBoost64 and Business specialization are separate candidates; no new CatBoost/specialist combination was tuned or selected.')
    configuration['business_augmented_xgb_weights']={'Business':{'xgb':.5,'business_specialist':.5},'Economy':{'xgb':1.}}
    configuration['business_equal_blend64_weights']={'Business':{'xgb':1/6,'business_specialist':1/6,'lgb':1/3,'cat_onehot64':1/3},
        'Economy':{'xgb':1/3,'lgb':1/3,'cat_onehot64':1/3}}
    save(REPO/'NEXT_MODEL_CONFIGURATIONS.json',configuration)
    d=pd.read_csv(direct/'analysis/primary/model_metrics.csv').set_index('model')
    b=pd.read_csv(business/'analysis/primary/model_metrics.csv').set_index('model')
    secondary=pd.read_csv(business/'analysis/sensitivity/model_metrics.csv').set_index('model') if (business/'analysis/sensitivity/model_metrics.csv').exists() else None
    rows=[('Frozen previous best blend','submission_equal_blend_onehot64.csv',d.loc['frozen_equal_blend64','r2']),
        ('Directly tuned CatBoost64','submission_catboost64_direct_tuned.csv',d.loc['cat64_direct','r2']),
        ('Blend with directly tuned CatBoost64','submission_equal_blend_direct64.csv',d.loc['direct_equal_blend64','r2']),
        ('Business-augmented XGBoost','submission_business_augmented_xgb.csv',b.loc['business_augmented_xgb','r2']),
        ('Blend with Business-augmented XGBoost','submission_business_equal_blend64.csv',b.loc['business_equal_blend64','r2'])]
    comparison=pd.DataFrame(rows,columns=['candidate','filename','primary_outer_r2']);comparison.to_csv(REPO/'NEXT_EXPERIMENT_COMPARISON.csv',index=False)
    def csv_link(filename):
        return 'https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/'+filename if filename=='submission_equal_blend_onehot64.csv' else 'submissions/'+filename
    table='\n'.join(['| Candidate | Primary outer R² | CSV |','| --- | ---: | --- |']+
        [f'| {name} | {score:.6f} | [{filename}]({csv_link(filename)}) |' for name,filename,score in rows])
    text=['# Follow-up airfare experiments — completed',
        'Two separate training-only experiments were executed: direct CatBoost64 tuning and a Business-only XGBoost specialist. The previous best is preserved. Four new comparison CSVs are in submissions/. These R² values are local training-only validation results, not Kaggle scores.',table]
    text.extend(['','Direct tuning did not improve the frozen CatBoost64 or its blend; its registered additional-seed compute gate was false. Those two CSVs are comparison candidates, not an evidence-based replacement for the previous best.',
        f'The Business-augmented XGBoost primary R² is {b.loc["business_augmented_xgb","r2"]:.6f}, versus {b.loc["global_xgb","r2"]:.6f} for global XGBoost. Its fixed three-family blend is {b.loc["business_equal_blend64","r2"]:.6f}, versus {b.loc["frozen_equal_blend64","r2"]:.6f} for the frozen blend. Inspect the paired folds: a pooled improvement does not mean every fold improved.'])
    if secondary is not None:
        text.append(f'The prescribed second split seed reran global tuning within each partition. Global XGBoost R²: {secondary.loc["global_xgb","r2"]:.6f}; Business-augmented XGBoost: {secondary.loc["business_augmented_xgb","r2"]:.6f}. This is standalone robustness evidence on reused observations, not a fresh holdout or a secondary blend result.')
    tails=pd.read_csv(business/'analysis/primary/tail_metrics.csv').set_index('model')
    text.append(f'The primary specialist trade-off is material: highest-priced-1% RMSE is {tails.loc["business_augmented_xgb","top1pct_target_rmse"]:,.2f}, versus {tails.loc["global_xgb","top1pct_target_rmse"]:,.2f} for global XGBoost. Mean training–validation R² gaps are {b.loc["business_augmented_xgb","mean_gap"]:.6f} and {b.loc["global_xgb","mean_gap"]:.6f}, respectively. A smaller average gap and better pooled R² do not imply improvement on the expensive tail.')
    text.extend(['## Exact fitting behavior',
        f'Original global XGBoost: {configs["xgb"]["selection"]["rounds"]} trees; original LightGBM: {configs["lgb"]["selection"]["rounds"]}; original CatBoost64: {configs["cat_onehot64"]["selection"]["rounds"]}; new directly tuned CatBoost64: {configs["cat64_direct"]["actual_trees"]}; new Business specialist: {configs["business_specialist"]["actual_trees"]}.',
        'Business-augmented XGBoost predicts Economy exactly as the original global XGBoost. On Business rows only it averages global XGBoost and the specialist 50/50. Its three-model blend averages that augmented prediction with original LightGBM and original CatBoost64 in fixed thirds. The directly tuned CatBoost64 is used only in the separate direct-CatBoost blend. No weights were optimized.',
        'The specialist trains only on 6,380 labeled Business rows, with the existing context features and a smaller bounded search. Its inner stopping metric evaluates the fixed halfblend, not outer labels. Pre-fit synthetic API checks resolved a documented XGBoost/scikit-learn estimator-tag compatibility issue without changing installed libraries.',
        'Full configuration dictionaries, resolved defaults, feature lists, hashes and weights are in NEXT_MODEL_CONFIGURATIONS.json. Per-fold selected settings and every trial are retained under each experiment directory.',
        '## Evidence and limitations',
        '[Direct CatBoost64 findings](airfare_round2/analysis/FINDINGS.md) and [Business-specialist findings](airfare_business/analysis/FINDINGS.md) report all folds, measured train–validation gaps, cabin/price-tail/rare-flight errors, stopping records and conditional group-bootstrap intervals. Blend training R² is not fabricated by averaging component scores.',
        'These folds and observations have informed prior development. The additional prescribed seed is a sensitivity check, not a pristine test. Bootstrap intervals omit retraining and accumulated selection uncertainty. Small differences can be noise; no absence of overfitting, Kaggle score improvement or rank is guaranteed. User-reported public results are provenance only and were not used as a tuning target.',
        'All five required full-data model bundles and both fixed prediction rules were frozen together before one final competition-test read. Four CSVs passed hashes, exact IDs/order, 3,898 rows, id/price schema and finite nonnegative-value checks. No models changed after test access and no Kaggle uploads were performed.',
        '## Run or inspect the code',
        'Use Python3.12 and install airfare_fresh/requirements.txt. Supply the original CSVs separately. The scripts use project-relative output folders and enforce source/input identities. Existing completed outputs resume or skip completed fits, rather than constituting new independent evidence. For a fresh rerun, work in a copy and archive its generated phase output directories first; preserve the original completed results and frozen controls.',
        '```bash\nexport OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1\nexport MPLCONFIGDIR=/tmp/airfare-mpl\npython tools/check_new_inference.py\npython airfare_round2/direct_cat64.py --stage all --train /path/to/train.csv\npython airfare_round2/analyze.py --train /path/to/train.csv\npython airfare_business/specialist.py --stage all --train /path/to/train.csv\npython airfare_business/analyze.py --train /path/to/train.csv\n```',
        'Final inference uses airfare_round2/inference.py with --include-business, --train, a new --out directory and --freeze-only first; then the same arguments plus --test and --sample. The supplied submissions directory already has an exclusive inference marker and refuses repeated access or overwrites. Do not remove the marker. Never tune using test predictions.',
        'The shared frozen source and original models remain in airfare_fresh/. Read each experiment PROTOCOL.md and REGISTRATION.json for prospective budgets, seeds and compute rules. No Business specialist was included in the earlier direct-tuning study’s budget; it is a separately registered experiment.'])
    (REPO/'NEXT_EXPERIMENTS.md').write_text('\n\n'.join(text)+'\n')
    ledger=dict(direct=read(direct/'analysis/completion.json'),business=read(business/'analysis/completion.json'),
        direct_compute_gate=read(direct/'sensitivity_decision.json'),business_compute_gate=read(business/'sensitivity_decision.json'),
        inference_audit=audit,primary_comparison=comparison.to_dict(orient='records'))
    ledger['total_research_model_fits']=ledger['direct']['research_model_fits']+ledger['business']['research_model_fits']
    ledger['total_trials']=ledger['direct']['trials_completed']+ledger['business']['total_trials']
    save(REPO/'NEXT_EXPERIMENT_LEDGER.json',ledger)
    files=[]
    for directory in [direct,business]:
        files.extend(p for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('-journal','-wal','-shm')))
    for p in files:
        if p.suffix=='.sqlite3':
            with sqlite3.connect(f'file:{p}?mode=ro',uri=True) as con:assert con.execute('pragma integrity_check').fetchone()[0]=='ok'
    save(REPO/'NEXT_EXPERIMENT_ARTIFACTS.json',{str(p.relative_to(REPO)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files})
    archive=REPO/'downloads/Airfare_Next_Experiments_Results.zip';assert not archive.exists()
    extras=[REPO/name for name in ['README.md','NEXT_EXPERIMENTS.md','NEXT_MODEL_CONFIGURATIONS.json','NEXT_EXPERIMENT_COMPARISON.csv','NEXT_EXPERIMENT_LEDGER.json','NEXT_EXPERIMENT_ARTIFACTS.json','NEXT_VALIDATION_AUDIT.json']]
    frozen_files=[p for p in sorted((REPO/'airfare_fresh').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('-journal','-wal','-shm'))]
    tools_files=[REPO/'tools/check_new_inference.py',REPO/'tools/audit_next_validation.py',Path(__file__)]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+extras+frozen_files+tools_files:z.write(p,str(p.relative_to(REPO)))
        for p in sorted((REPO/'submissions').glob('*.csv')):z.write(p,'submissions/'+p.name)
    small=REPO/'downloads/Airfare_New_Submissions_and_Configs.zip';assert not small.exists()
    with zipfile.ZipFile(small,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for filename in audit['submission_sha256']:z.write(REPO/'submissions'/filename,'submissions/'+filename)
        for p in extras[1:]:z.write(p,p.name)
        for folder in [direct,business]:z.write(folder/'analysis/FINDINGS.md',folder.name+'/FINDINGS.md')
    for p in [archive,small]:
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        print(p.name,p.stat().st_size,'bytes SHA256',sha(p))


if __name__=='__main__':main()
