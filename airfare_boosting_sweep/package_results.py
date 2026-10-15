"""Package completed outputs; never read competition train/test/sample inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    audit=read(ROOT/'submissions/inference_audit.json')
    assert audit['output_count']==10 and audit['test_content_reads']==1
    assert not audit['model_changes_after_test_access']
    configs=read(ROOT/'MODEL_CONFIGURATIONS.json')
    table=pd.read_csv(ROOT/'analysis/model_metrics.csv')
    singles=configs['models'];blends=configs['blends']
    rows=[]
    for name,c in singles.items():
        metric=table.set_index('model').loc[name]
        rows.append(dict(model=name,trees=c['actual_rounds'],requested=c['selection']['configuration']['rounds'],
            cv_r2=float(metric.r2),train_gap=float(metric.mean_gap),file=f'submission_highround_optuna_{name}.csv'))
    for name,weights in blends.items():
        metric=table.set_index('model').loc[name]
        rows.append(dict(model=name,trees=' + '.join(str(singles[k]['actual_rounds']) for k in weights),
            requested='fixed equal thirds',cv_r2=float(metric.r2),train_gap=float(metric.mean_gap),
            file=f'submission_highround_optuna_{name}.csv'))
    def markdown(rows):
        lines=['| Candidate | Actual trees / rounds | Outer CV R² | Mean train–validation gap | CSV |',
               '| --- | ---: | ---: | ---: | --- |']
        for row in sorted(rows,key=lambda r:r['cv_r2'],reverse=True):
            lines.append(f"| {row['model']} | {row['trees']} | {row['cv_r2']:.6f} | {row['train_gap']:.6f} | [{row['file']}](submissions/{row['file']}) |")
        return '\n'.join(lines)
    text=['# Broad high-round Optuna boosting study',
        'Eight global boosting procedures and two predeclared equal-weight blends, executed in response to the request for multiple aggressive candidates. Every successful candidate is supplied, including weaker comparisons. These are measured training-only results; no new public/private Kaggle R² or rank is claimed.',
        markdown(rows),
        'Same-fold historical controls: '+', '.join(f'{row.model}: R²={row.r2:.6f}' for row in table.loc[table.model.str.startswith('control_')].itertuples(index=False))+'.',
        '[The saved-tree audit](TREE_COMPLEXITY_AUDIT.json) checks actual fitted counts and leaf bounds. Its structural inspection performs no model fitting or test-data access.',
        'Every Optuna trial requested 800, 1100 or 1400 rounds. Actual fitted counts are shown above; AdaBoost or a library without further useful splits can terminate before its requested budget. Symmetric CatBoost’s leaf cap is implicit in its finite depth; other tree depths/leaves/support/regularization are explicitly bounded where supported.',
        '## Evidence and exact configurations',
        '- [Full measured findings](analysis/FINDINGS.md)\n- [Exact final model dictionaries, resolved parameters and blend weights](MODEL_CONFIGURATIONS.json)\n- [All 336 trial records](analysis/all_trials.csv)\n- [Paired outer fold metrics](analysis/fold_metrics.csv)\n- [Tail errors](analysis/tail_metrics.csv)\n- [Cabin, prefix and rare-flight diagnostics](analysis/subgroup_metrics.csv)\n- [Grouped-bootstrap uncertainty](analysis/bootstrap_intervals.json)\n- [Partition/source identity audit](VALIDATION_AUDIT.json)\n- [Fit accounting](EXPERIMENT_LEDGER.json)\n- [Frozen inference manifest](submissions/frozen_manifest.json)\n- [Submission schema/ID/checksum audit](submissions/inference_audit.json)',
        'A parallel-scheduling collision required one identical-configuration LightGBM trial recovery. Actual accounting is 336 distinct Optuna trials, 384 canonical research fits and one duplicate fit attempt (385 total), plus 15 API smoke fits. See SCHEDULING_RECOVERY.json and EXPERIMENT_LEDGER.json. No search-space expansion or outer/test-based choice was introduced. Saved parameters preserve the exact asynchronous TPE history for the affected scope.',
        '## Validation meaning',
        'Five outer grouped/stratified folds evaluate each whole tuning procedure. Inside each outer fitting partition, a separate inner 80/20 group split ranks the fixed-budget Optuna trials. Outer validation labels never choose parameters or counts. Full-data settings come from a fresh same-budget inner tuning run, then a refit on all 15589 labeled rows. All feature mappings fit only on the corresponding fitting partition. The unchanged prior global and Business-augmented controls use identical outer rows/folds.',
        'These folds and training observations have already informed earlier development. There is no pristine new holdout. Single-inner-split tuning is an efficiency compromise, and comparing many families adds selection uncertainty. Confidence intervals condition on saved predictions. A high count or high training R² does not ensure a public gain. The previous user-reported best 0.95881 remains available; this study neither uses public scores to tune settings nor verifies final-selection rules.',
        '## Reproduce',
        'From a fresh checkout/copy without this phase’s generated outputs, create a Python3.12 environment and install the pinned dependencies in ../airfare_fresh/requirements.txt. Set OMP_NUM_THREADS=4, OMP_WAIT_POLICY=PASSIVE, OPENBLAS_NUM_THREADS=1 and MPLCONFIGDIR to a writable directory. All commands below run from the repository root. Supply your own original competition data paths.',
        '```bash\npython airfare_boosting_sweep/study.py --stage smoke --train /path/train.csv\npython airfare_boosting_sweep/study.py --stage register --train /path/train.csv\npython airfare_boosting_sweep/study.py --stage run --train /path/train.csv\npython airfare_boosting_sweep/analyze.py --train /path/train.csv\npython airfare_boosting_sweep/check_inference.py\npython airfare_boosting_sweep/inference.py --freeze-only --train /path/train.csv\npython airfare_boosting_sweep/inference.py --train /path/train.csv --test /path/test.csv --sample /path/sample_submission.csv\n```',
        'The supplied generated experiment is frozen and inference has already completed. Do not delete its markers or overwrite artifacts to rerun it. Reproduction needs a fresh output copy without generated registration/results/analysis/audits/submissions; keep all source files and sibling dependencies. The original training/test/sample SHA256 checks protect input identity. Runtime parallel_*.py and recover_scheduling.py helpers preserve this machine’s scheduling history and process IDs; use the documented sequential study.py workflow on another machine. Serialized joblib files should be loaded only from a trusted source.',
        '## Downloads',
        'The top-level submissions folder contains byte-identical copies. downloads/Airfare_HighRound_Submissions_and_Configs.zip contains the ten CSVs, exact configurations, code and diagnostics. downloads/Airfare_HighRound_Complete.zip additionally contains the eight fitted model bundles and trial databases. Sibling airfare_fresh source/dependencies are included in both archives. Raw competition datasets and the virtual environment are excluded.']
    (ROOT/'README.md').write_text('\n\n'.join(text)+'\n')
    top=REPO/'submissions';top.mkdir(exist_ok=True)
    for filename,expected in audit['files'].items():
        source=ROOT/'submissions'/filename;assert sha(source)==expected
        target=top/filename
        if target.exists():assert sha(target)==expected
        else:shutil.copyfile(source,target)
    files={}
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='ARTIFACT_MANIFEST.json':
            files[str(path.relative_to(ROOT))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    save(ROOT/'ARTIFACT_MANIFEST.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        files=files,note='Excludes this manifest and interpreter caches.'))
    downloads=REPO/'downloads';downloads.mkdir(exist_ok=True)
    dependencies=[REPO/'airfare_fresh/study.py',REPO/'airfare_fresh/features.py',
                  REPO/'airfare_fresh/diagnostics.py',REPO/'airfare_fresh/requirements.txt',
                  REPO/'airfare_fresh/analysis/aligned_training_oof.csv',
                  REPO/'airfare_fresh/eda/TRAIN_ONLY_EDA.md',
                  REPO/'airfare_business/primary/global_xgb_oof.csv',
                  REPO/'airfare_business/primary/business_augmented_xgb_oof.csv']
    dependencies += [REPO/'airfare_business/primary'/f'outer{i}/outer_metrics.json' for i in range(5)]
    for name,include_models in [('Airfare_HighRound_Submissions_and_Configs.zip',False),('Airfare_HighRound_Complete.zip',True)]:
        target=downloads/name
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in sorted(ROOT.rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts:continue
                if not include_models and (path.suffix in ['.joblib','.sqlite3','.npz']):continue
                archive.write(path,path.relative_to(REPO))
            for path in dependencies:archive.write(path,path.relative_to(REPO))
        with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
        print(name,target.stat().st_size,'bytes',sha(target),flush=True)
    print(markdown(rows),flush=True)


if __name__=='__main__':main()
