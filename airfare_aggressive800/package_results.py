"""Publish completed candidate artifacts; no fitting or test-feature analysis."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    out=ROOT/'submissions';audit=read(out/'inference_audit.json');freeze=read(out/'frozen_manifest.json')
    assert sha(out/'frozen_manifest.json')==audit['manifest_sha256']
    for name,digest in freeze['snapshot']['source_hashes'].items():assert sha(REPO/name)==digest,name
    filename=audit['filename'];source=out/filename;assert sha(source)==audit['submission_sha256']
    candidate=pd.read_csv(source,dtype={'id':str})
    ids=pd.read_csv(REPO/'submissions/submission_business_augmented_xgb.csv',usecols=['id'],dtype={'id':str}).id
    assert len(candidate)==3898 and list(candidate)==['id','price'] and candidate.id.equals(ids) and candidate.id.is_unique
    assert np.isfinite(candidate.price).all() and (candidate.price>=0).all()
    target=REPO/'submissions'/filename;assert not target.exists() or target.read_bytes()==source.read_bytes()
    shutil.copy2(source,target)
    save(ROOT/'model_configurations.json',freeze['snapshot'])
    primary=pd.read_csv(ROOT/'analysis/primary/model_metrics.csv').set_index('model')
    secondary=pd.read_csv(ROOT/'analysis/sensitivity/model_metrics.csv').set_index('model')
    rows=[]
    for label,df in [('primary',primary),('sensitivity',secondary)]:
        summary=read(ROOT/label/'summary.json')
        rows.append(dict(validation=label,control_r2=float(df.loc['current_business_augmented','r2']),
            aggressive_r2=float(df.loc['aggressive800','r2']),control_mean_gap=float(df.loc['current_business_augmented','mean_gap']),
            aggressive_mean_gap=float(df.loc['aggressive800','mean_gap']),paired_fold_wins=sum(r['delta_r2']>0 for r in summary['folds'])))
    pd.DataFrame(rows).to_csv(ROOT/'comparison.csv',index=False)
    save(ROOT/'experiment_ledger.json',dict(registration=read(ROOT/'REGISTRATION.json'),completion=read(ROOT/'analysis/completion.json'),
        comparison=rows,inference_audit=audit,prior_public_result={'source':'User reported; not independently verified','business_augmented_xgb_r2':.95881},
        candidate_public_result=None))
    table='\n'.join(['| Reused split seed | Control R² | Aggressive R² | Control mean gap | Aggressive mean gap | Paired fold wins |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']+
        [f'| {r["validation"]} | {r["control_r2"]:.6f} | {r["aggressive_r2"]:.6f} | {r["control_mean_gap"]:.6f} | {r["aggressive_mean_gap"]:.6f} | {r["paired_fold_wins"]}/5 |' for r in rows])
    text=['# One exploratory 800-tree Business candidate',
        f'**[Download the single new CSV](https://github.com/devshah207/Domestic_Flight_Prediction_Challenge/raw/refs/heads/main/submissions/{filename})**',
        'This is the one higher-capacity candidate requested for an exploratory portfolio slot. No other new submission files were created. Prior submissions remain unchanged. The earlier Business-augmented model’s public R² of 0.95881 is user-reported; this new candidate has no known Kaggle score.',
        table,
        'Pooled validation R² improved on both prescribed split seeds, while the training–validation gaps increased substantially. The primary gain is concentrated in two folds; all five secondary folds improved. Higher capacity captured additional validation signal but also increased fitting flexibility. This does not establish a top-three public rank, private-score improvement or absence of overfitting.',
        '## Exact prediction model',
        'The new Business specialist has **800 trees**, max_depth=7, max_leaves=48, learning_rate=0.045, min_child_weight=8, reg_lambda=3, reg_alpha=0.05, gamma=0.01, subsample=0.95 and colsample_bytree=0.90. Hist/depthwise/native categorical processing, max_bin=192, max_cat_threshold=32, max_cat_to_onehot=4, random_state=42 and four threads remain fixed. These settings were registered together before any new scores, so differences cannot be attributed to tree count alone.',
        'The global XGBoost remains the original 454-tree model. Economy uses that global prediction unchanged. Business uses exactly 50% global plus 50% new specialist. Thus the CSV represents one composite predictor with an 800-tree Business branch. Targets train as price/10000; each component is inverted and clipped below at zero before mixing.',
        'One fixed profile, zero Optuna trials, zero stopping-count searches, 21 research fits: ten reconstructed global outer models, ten new Business outer models and one full Business model trained on 6,380 rows. Final global weights were reused without retraining.',
        '## Validation evidence',
        'Read [analysis/FINDINGS.md](analysis/FINDINGS.md) for original-price R²/RMSE/MAE, measured composite training gaps, paired folds, high-price errors and conditional group-bootstrap intervals. Subgroup/worst-error CSVs and plots are under analysis/. Both primary and secondary splits reuse observations and folds from development; neither is a pristine holdout. Bootstrap intervals omit retraining and accumulated selection uncertainty.',
        'Global controls are reconstructed from partition-local settings and verified against saved OOF predictions. Stored raw/target identities and group separation are checked before fitting. New feature mappings learn only from each outer Business-training subset. Outer labels never choose this fixed profile, tree count or weights. No competition test features or predictions informed development.',
        '## Files and reproduction',
        'model_configurations.json contains both exact fitted model dictionaries, resolved booster settings, versions, hashes, feature lists and the fixed prediction policy. PROTOCOL.md and REGISTRATION.json record the prospective recipe and budget. experiment_ledger.json records actual completed results and inference audit. ARTIFACT_MANIFEST.json covers all candidate files except itself.',
        'Clone the complete GitHub repository to run the code: source dependencies and frozen controls live in sibling airfare_business/, airfare_round2/ and airfare_fresh/ folders. Install airfare_fresh/requirements.txt with Python 3.12 and supply the original CSVs separately. The candidate archive packages its source, model, diagnostics and one submission; it requires those sibling dependencies.',
        '```bash\nexport OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1\nexport MPLCONFIGDIR=/tmp/airfare-mpl\npython airfare_aggressive800/check_inference.py\npython airfare_aggressive800/experiment.py --stage all --train /path/to/train.csv\n```',
        'Completed fitting scopes resume from verified existing artifacts. For a fresh rerun work in a copy, archive the generated primary/, sensitivity/, full_training/, analysis/ and submissions/ folders there, and preserve the original results. The source/input hash guards require unchanged inputs and source contents.',
        'Final inference uses inference.py --train PATH --out NEW_DIRECTORY --freeze-only, then the same arguments plus --test PATH --sample PATH. Do not reuse the supplied submissions/ directory: its exclusive inference marker prevents repeated access or overwrites. Models were frozen before this phase’s one test-content read; no model changes followed.',
        'The single CSV passed exact sample-ID ordering, id/price schema, 3,898 rows, unique IDs and finite nonnegative-value checks. No Kaggle submission was uploaded. Keep the earlier candidate files; the competition’s actual final-selection allowance remains unverified.']
    (ROOT/'README.md').write_text('\n\n'.join(text)+'\n')
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('-journal','-wal','-shm'))]
    save(ROOT/'ARTIFACT_MANIFEST.json',{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files})
    files.append(ROOT/'ARTIFACT_MANIFEST.json')
    archive=REPO/'downloads/Airfare_Aggressive800_Candidate.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(REPO)))
        z.write(target,'submissions/'+filename)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('One CSV verified and copied:',target)
    print('Archive:',archive.name,archive.stat().st_size,'bytes SHA256',sha(archive))


if __name__=='__main__':main()
