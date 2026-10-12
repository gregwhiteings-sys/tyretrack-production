"""TyreTrack V5 report-content dry run. Never sends emails or touches production DB.

Windows-safe: helper functions in the application may briefly retain SQLite file
handles. Cleanup is best-effort and cannot turn passing assertions into an error.
"""
import ast
import gc
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE = Path(__file__).with_name('TyreTrack_Production_Monitor.py')
code = SOURCE.read_text(encoding='utf-8')
compile(code, str(SOURCE), 'exec')
mod = ast.parse(code)
names = {'tyre_class_name', 'production_report_counts', 'build_hourly_report'}
selected = [n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
missing = names - {n.name for n in selected}
if missing:
    raise RuntimeError(f'Missing required functions in application: {sorted(missing)}')
namespace = {
    'sqlite3': sqlite3,
    'datetime': datetime,
    'timedelta': timedelta,
    'timezone': timezone,
    'OAK_NAME': 'OAK-D S2 PoE / CAT5',
}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)

BRISBANE = timezone(timedelta(hours=10))
def at(day, h, m=0):
    return datetime(2026, 10, day, h, m, tzinfo=BRISBANE)

checks = []
def check(label, condition):
    checks.append(bool(condition))
    print(('PASS' if condition else 'FAIL') + ': ' + label)

# mkdtemp rather than TemporaryDirectory: Windows may briefly retain SQLite
# handles opened by application helpers. Cleanup is retried and never masks tests.
temp_dir = Path(tempfile.mkdtemp(prefix='tyretrack_report_test_'))
try:
    db = temp_dir / 'synthetic.db'
    namespace['DB'] = db
    conn = sqlite3.connect(db)
    try:
            conn.execute('CREATE TABLE counts(id INTEGER PRIMARY KEY,ts TEXT,source TEXT,track_id INTEGER,confidence REAL,tyre_type TEXT,class_confidence REAL,metric_utc REAL)')
            def insert(stamp, kind='passenger', source='OAK-D S2 PoE / CAT5'):
                conn.execute('INSERT INTO counts(ts,source,track_id,confidence,tyre_type,class_confidence,metric_utc) VALUES(?,?,?,?,?,?,?)', (stamp.isoformat(), source, 1, .9, kind, .9, stamp.timestamp()))
            insert(at(12, 5, 10), 'passenger')
            insert(at(12, 5, 15), '4x4')
            insert(at(12, 5, 30), 'unknown')
            insert(at(12, 5, 31), 'passenger', source='video')
            insert(at(12, 6, 1), 'suv')
            insert(at(12, 16, 59), 'passenger')
            insert(at(12, 17, 2), '4x4')
            insert(at(12, 17, 6), 'passenger')
            insert(at(11, 9, 0), 'passenger')
            conn.commit()
    finally:
        conn.close()


    r = namespace['production_report_counts'](at(12, 5), at(12, 6))
    check('05:00–06:00 hourly count is 3', r['hourly'] == 3)
    check('Hourly classes reconcile 1+1+1=3', r['hourly_types'] == {'passenger': 1, '4x4': 1, 'unknown': 1})
    check('Video source excluded', r['daily'] == 3)
    check('Previous day excluded from daily', r['daily'] == 3)
    check('All-time includes only OAK records', r['all_time'] == 8)
    s, body = namespace['build_hourly_report'](at(12, 5), at(12, 6))
    check('Report matches database hourly count', 'OAK HOURLY TOTAL: 3' in body)
    check('Report matches database daily count', 'OAK DAILY TOTAL: 3' in body)
    check('Report includes hourly and daily breakdowns', body.count('Passenger: 1') == 2)
    check('Hourly flow rate uses elapsed period', 'AVERAGE FLOW: 0.05 tyres/min' in body)
    check('Brisbane timezone stated', 'Brisbane time' in body)
    r2 = namespace['production_report_counts'](at(12, 6), at(12, 7))
    check('06:00–07:00 hour contains one tyre', r2['hourly'] == 1 and r2['daily'] == 4)
    r3 = namespace['production_report_counts'](at(12, 17), at(12, 17, 5))
    check('Final 5-minute slot contains one tyre', r3['hourly'] == 1 and r3['daily'] == 6)
    check('17:06 excluded from final period', r3['daily'] == 6)
    _, last = namespace['build_hourly_report'](at(12, 17), at(12, 17, 5))
    check('Final-slot flow rate divides by 5 minutes', 'AVERAGE FLOW: 0.20 tyres/min' in last)
    check('Daily classification totals reconcile', sum(r3['daily_types'].values()) == r3['daily'])
    check('Hourly classification totals reconcile', sum(r3['hourly_types'].values()) == r3['hourly'])
finally:
    # Release objects that may still reference temporary SQLite connections.
    gc.collect()
    for attempt in range(5):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            if attempt < 4:
                gc.collect()
                time.sleep(0.3)
            else:
                print(f'NOTE: Windows still has a temporary test DB open: {temp_dir}')
                print('This does not affect test results or production data.')

print(f'\n{sum(checks)}/{len(checks)} checks passed')
if not all(checks):
    raise SystemExit(1)
