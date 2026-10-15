"""Report saved research progress without importing ML libraries or reading data."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent


def main():
    reg=json.loads((ROOT/'REGISTRATION.json').read_text());finished=set()
    for path in ROOT.glob('*.log'):
        for line in path.read_text().splitlines():
            if line.startswith('TRIAL '):finished.add(tuple(line.split()[1:4]))
    recovery=ROOT/'SCHEDULING_RECOVERY.json'
    if recovery.exists():
        record=json.loads(recovery.read_text())
        if record.get('recovered_trial_complete'):
            finished.add((record['family'],record['scope'],str(record['trial'])))
    print('family                    trial fit records   outer folds   final actual rounds')
    for family in reg['families']:
        folder=ROOT/'results'/family
        trials=sum(key[0]==family for key in finished);budget=reg['trials_per_scope'][family]*6
        outer=len(list(folder.glob('outer*/metrics.json')))
        path=folder/'full_training/configuration.json'
        final=json.loads(path.read_text())['actual_rounds'] if path.exists() else 'pending'
        print(f'{family:25} {trials:3}/{budget:<3}              {outer}/5           {final}')
    print(f'Finished trial fit records: {len(finished)}/{reg["research_trial_budget"]}. Logs are progress indicators; final audit verifies persisted trial states.')


if __name__=='__main__':main()
