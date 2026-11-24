"""Record the user's stop instruction without evolving or rescoring any state."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / 'CN_REVISION_20261118'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    receipt = HERE / 'USER_STOP_RECEIPT.json'
    if receipt.exists():
        print('Stop instruction already recorded; original receipt preserved.')
        return
    now = datetime.datetime.now().astimezone().isoformat()
    before = read(HERE / 'CAMPAIGN_STATUS.json')
    inventory = []
    for path in sorted(HERE.rglob('STATUS.json')):
        original = read(path)
        record = dict(source=str(path.relative_to(HERE)), original_status=original,
                      original_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if original.get('status') == 'RUNNING':
            record['disposition'] = 'STOPPED_BY_USER_INCOMPLETE_EXCLUDED_FROM_PAIRED_SCORES'
            stopped = {**original, 'status': 'STOPPED_BY_USER',
                       'stopped_at': now, 'reason': 'User ended additional validation at the current scope.'}
            write(path, stopped)
        inventory.append(record)
    data = dict(status='STOPPED_BY_USER', updated=now,
                instruction='到目前为止可以了，我觉得参考数据精度不用太严格，反正是对比方法',
                process_check='Escalated Get-Process python* and Win32_Process inspections found no Python validation or watcher processes before freezing.',
                previous_campaign_status=before, trajectories=inventory,
                formal_report=read(HERE / 'LATEST_REPORT.json'),
                policy='Use complete independently audited pairs against a common discrete reference; preserve historical qualification failures; no new simulations or inference.',
                original_production_data_modified=False)
    write(receipt, data)
    write(HERE / 'CAMPAIGN_STATUS.json', {**before, 'status': 'STOPPED_BY_USER',
          'stopped_at': now, 'current': None, 'interrupted_task': before.get('current'),
          'stop_receipt': receipt.name, 'automatic_resume_authorized': False})
    write(HERE / 'POSTPROCESS_STATUS.json', dict(status='STOPPED_BY_USER', updated=now,
          campaign_still_active=False, watcher_active=False, automatic_resume_authorized=False))
    write(STAGE / 'audit' / receipt.name, data)
    print(json.dumps(dict(status='STOPPED_BY_USER', trajectory_status_files=len(inventory),
                         formal_report=data['formal_report']), ensure_ascii=False))

if __name__ == '__main__':
    main()
