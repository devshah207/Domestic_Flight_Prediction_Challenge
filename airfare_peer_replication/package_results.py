"""Package frozen peer replication and its seven audited CSVs, no data/model fits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    audit=read(ROOT/'submissions/inference_audit.json')
    assert audit['output_count']==7 and audit['test_content_reads']==1 and not audit['models_changed_after_test']
    config=read(ROOT/'MODEL_CONFIGURATIONS.json');table=pd.read_csv(ROOT/'analysis/model_metrics.csv').set_index('model')
    lines=['# Replication of the disclosed RF + XGBoost + CatBoost ensemble','',
        'Two equal-weight blends reproduce the disclosed model settings with explicit preprocessing assumptions. The `smooth="auto"` clue is interpreted as scikit-learn TargetEncoder; its exact use by the peer is unconfirmed. This does not claim an exact full-pipeline replication or reproduction of their reported 0.96080.','',
        '| Model / file | Trees | Max depth | Learning rate | Training-only outer R² | Mean train–validation gap |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,v in config['models'].items():
        params=v['constructor'];s=v['actual_structure'];r=table.loc[name]
        lr=params.get('learning_rate','—');filename=f'submission_peer_{name}.csv'
        lines.append(f"| [{name}](submissions/{filename}) | {s['actual_trees']} | {s['configured_depth_bound']} | {lr} | {r.r2:.6f} | {r.mean_gap:.6f} |")
    for name,weights in config['blends'].items():
        r=table.loc[name]
        lines.append(f"| [{name}](submissions/submission_peer_{name}.csv) | 200 + 964 + 1000 | 2 / 9 / 6 | — | {r.r2:.6f} | {r.mean_gap:.6f} |")
    lines+=['',
        'RF uses all fitting rows in every tree (`bootstrap=False`), 70% feature sampling per split and seed42. XGB uses 964 trees, depth9, lr0.054 and seed42. CatBoost uses 1000 trees, depth6, lr0.139 and seed42. Both blends are fixed equal arithmetic thirds. Unknown peer parameters are documented explicitly in [the protocol](PROTOCOL.md); every constructor, resolved library setting, actual structure and feature schema is in [MODEL_CONFIGURATIONS.json](MODEL_CONFIGURATIONS.json).','',
        '`equal_smooth_auto` uses smoothed categorical target encodings for RF and XGB and native categorical CatBoost. Each encoding training row is produced by a mapping that excludes its entire itinerary group. A separate map fit only on the corresponding full fitting partition encodes held-out validation/test rows; unseen categories fall back to the fitting target mean. `equal_ohe` changes RF and XGB to training-fitted sparse one-hot encoding and uses the identical CatBoost. Both use the existing compact context feature set as an assumption. No ID or validation prices enter the features.','',
        'Five training-only grouped/stratified folds evaluated fixed configurations. There was no Optuna search, early stopping, weight optimization or selection of a winning encoding from these scores. Thirty regressor fits completed (25 outer +5 full), plus72 research TargetEncoder mapping fits. Twelve additional encoder fits were synthetic leakage checks; no smoke regressors were fitted. These folds and observations have informed earlier development and are not a pristine holdout. Outer R² is not a public/private Kaggle score.','',
        'The shallow RF may materially weaken an equal blend; all components and both registered blends are delivered to make this fixed replication reviewable. Their gaps, tail errors and cabin errors are measured rather than presented as guarantees. The earlier user-reported public best remains 0.95885 from classic gradient boosting; no new public score is known for this phase.','',
        '## Evidence','',
        '- [Training-only findings and historical controls](analysis/FINDINGS.md)\n- [Fold metrics](analysis/fold_metrics.csv)\n- [Tail errors](analysis/tail_metrics.csv)\n- [Cabin errors](analysis/cabin_metrics.csv)\n- [Residual correlations](analysis/residual_correlations.csv)\n- [Exact configurations](MODEL_CONFIGURATIONS.json)\n- [Fit ledger](EXPERIMENT_LEDGER.json)\n- [Synthetic leakage and inference checks](synthetic_checks.json)\n- [Frozen source/model/policy manifest](submissions/frozen_manifest.json)\n- [Completed inference checks and CSV hashes](submissions/inference_audit.json)','',
        '## Reproduce','',
        'Use Python3.12 and the pins in ../airfare_fresh/requirements.txt. Start from a fresh output copy without generated registration/results/analysis/submissions/audits; preserve all source files and sibling source dependencies. The delivered experiment is already frozen: do not remove its one-use inference marker or overwrite it. Provide your own original competition CSVs. From the repository root:',
        '```bash\nexport OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1\nexport MPLCONFIGDIR=/tmp/airfare-mpl\npython airfare_peer_replication/check_inference.py\npython airfare_peer_replication/run.py --stage register --train /path/train.csv\npython airfare_peer_replication/run.py --stage train --train /path/train.csv\npython airfare_peer_replication/inference.py --freeze-only --train /path/train.csv\npython airfare_peer_replication/inference.py --train /path/train.csv --test /path/test.csv --sample /path/sample_submission.csv\n```','',
        'Both encodings, all five full models, source/version hashes, schemas, price scaling and blend weights were frozen together before this phase read test content once. Seven files passed 3898 rows, exact sample ID order, id/price columns and finite nonnegative prices. No model fitting or revision followed test access. The top-level submissions folder contains byte-identical copies. No Kaggle upload was performed. Only load serialized models from a trusted source.','',
        'downloads/Airfare_Peer_Replication_Submissions.zip includes seven CSVs, code, exact configurations and diagnostics. downloads/Airfare_Peer_Replication_Complete.zip also includes the fitted model bundles and cross-fitted training prediction arrays. Archives exclude raw competition datasets and the virtual environment.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    top=REPO/'submissions'
    for filename,checksum in audit['files'].items():
        path=ROOT/'submissions'/filename;assert sha(path)==checksum
        target=top/filename
        if target.exists():assert sha(target)==checksum
        else:shutil.copyfile(path,target)
    files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json'}
    save(ROOT/'ARTIFACT_MANIFEST.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),files=files))
    deps=[REPO/'airfare_fresh'/n for n in ['features.py','study.py','requirements.txt','analysis/aligned_training_oof.csv']]
    deps+=[REPO/'airfare_business/primary/business_augmented_xgb_oof.csv']
    for name,complete in [('Airfare_Peer_Replication_Submissions.zip',False),('Airfare_Peer_Replication_Complete.zip',True)]:
        target=REPO/'downloads'/name
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(ROOT.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts:continue
                if not complete and p.suffix in ['.joblib','.npz']:continue
                z.write(p,p.relative_to(REPO))
            for p in deps:z.write(p,p.relative_to(REPO))
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        assert target.stat().st_size<100_000_000
        print(name,target.stat().st_size,'bytes',sha(target),flush=True)
    print('\n'.join(lines[4:15]),flush=True)

if __name__=='__main__':main()
