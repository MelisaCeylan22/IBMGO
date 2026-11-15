"""Ayrı kayıtlarla, kesintiden devam edebilen ek IBMGO deneyleri.

python -B -m kod.deneyler.ek_deneyler --experiment iteration --output sonuclar/ek_iterasyon_2026-11-13
python -B -m kod.deneyler.ek_deneyler --experiment scale --output sonuclar/ek_olcek_2026-11-13
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from kod.algoritmalar import baselines
from kod.algoritmalar.bmgo import binary_mountain_gazelle_optimizer
from kod.algoritmalar.fitness import calculate_fitness, get_suite_metrics
from kod.deneyler.sunum_deneyi import ROOT, Settings, digest_pool, make_pool, validate_pool


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path, frame):
    temp = path.with_suffix('.csv.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, path)


@contextmanager
def experiment_lock(folder):
    """İşlem kapanınca işletim sisteminin bıraktığı, aynı klasöre tek yazar kilidi."""
    with (folder / 'experiment.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Bu deney klasöründe başka bir çalıştırma sürüyor.') from exc
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def experiment_plan(kind, smoke=False):
    if kind not in {'iteration', 'scale'}:
        raise ValueError('Deney türü iteration veya scale olmalı.')
    sizes = [250, 500] if kind == 'iteration' else [2000, 5000]
    data_seeds = [83, 97, 109]
    algorithm_seeds = list(range(42, 47 if kind == 'iteration' else 45))
    iterations = [20, 50, 100] if kind == 'iteration' else [20]
    if smoke:
        sizes, data_seeds, algorithm_seeds = [50], [83], [42]
        iterations = [2, 3] if kind == 'iteration' else [2]
    methods = ['IBMGO'] if kind == 'iteration' else ['IBMGO', 'GA', 'Greedy']
    cases = []
    for size in sizes:
        for data_seed in data_seeds:
            for method in methods:
                for seed in ([None] if method == 'Greedy' else algorithm_seeds):
                    for steps in iterations:
                        case_id = f'{size}_{data_seed}_{method}_{seed}_{steps}'
                        cases.append(dict(case_id=case_id, size=size, data_seed=data_seed,
                                          algorithm_seed=seed, method=method, iterations=steps))
    return dict(kind=kind, smoke=smoke, sizes=sizes, data_seeds=data_seeds,
                algorithm_seeds=algorithm_seeds, iterations=iterations, methods=methods,
                population=12, local_trials=12, cases=cases)


def code_identity():
    sources = (sorted((ROOT / 'kod/algoritmalar').glob('*.py'))
               + sorted((ROOT / 'kod/veri').glob('*.py'))
               + [Path(__file__), ROOT / 'kod/deneyler/sunum_deneyi.py'])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    return dict(hashes=hashes, git_commit=git.stdout.strip(),
                working_tree=dirty.stdout.splitlines(), python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform())


def protected_identity():
    files = [ROOT / 'IBMGO_Sunum.ipynb']
    files += sorted((ROOT / 'sonuclar').glob('dayaniklilik_*/**/*'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def pool_structure(size, seed, pool):
    support = np.count_nonzero(pool['coverage'], axis=0)
    return dict(test_sayisi=size, veri_seed=seed, kapsam_hedefi=len(support),
                kritik_hedef=len(pool['critical']), on_kosul_baglantisi=int(pool['dependencies'].sum()),
                hedef_destegi_min=int(support.min()), hedef_destegi_medyan=float(np.median(support)),
                hedef_destegi_max=int(support.max()), girdi_kimligi=digest_pool(pool))


def prepare(folder, plan):
    pools = {(n, seed): make_pool(n, seed) for n in plan['sizes'] for seed in plan['data_seeds']}
    for pool in pools.values():
        validate_pool(pool, .9)
    version = code_identity()
    identity = dict(plan=plan, code_hashes=version['hashes'], python=version['python'],
                    numpy=version['numpy'], pandas=version['pandas'],
                    inputs={f'{n}_{seed}': digest_pool(p) for (n, seed), p in pools.items()})
    fingerprint = digest_json(identity)
    path = folder / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Devam edilemiyor. Deney planı, hesaplama kodu, kütüphane sürümü veya girdi kimliği değişmiş.')
        if manifest['protected_files'] != protected_identity():
            raise ValueError('Ana sunum veya ana arşiv değişmiş. Kayıtlı deney kimliğini kontrol edin.')
    else:
        manifest = dict(fingerprint=fingerprint, identity=identity, version=version,
                        protected_files=protected_identity(), created_utc=timestamp(),
                        status='prepared', completed=0, total=len(plan['cases']))
        atomic_json(path, manifest)
    (folder / 'girdiler').mkdir(exist_ok=True)
    (folder / 'kosumlar').mkdir(exist_ok=True)
    for (n, seed), pool in pools.items():
        target = folder / 'girdiler' / f'{n}_{seed}.npz'
        if target.exists():
            with np.load(target) as saved:
                actual = dict(saved)
            if digest_pool(actual) != digest_pool(pool):
                raise ValueError(f'Kayıtlı girdi değişmiş veya bozulmuş. {n} test, veri {seed}.')
        else:
            temp = target.with_suffix('.npz.tmp')
            with temp.open('wb') as handle:
                np.savez_compressed(handle, **{k: pool[k] for k in ['coverage', 'durations', 'dependencies', 'critical']})
            os.replace(temp, target)
    atomic_csv(folder / 'havuz_yapisi.csv', pd.DataFrame([
        pool_structure(n, seed, pool) for (n, seed), pool in pools.items()
    ]))
    return pools, manifest


def verify_trace(trace, curve, final, planned):
    events = trace['events']
    assert trace['evaluations'] == len(events) <= planned
    assert [event['evaluation'] for event in events] == list(range(1, len(events) + 1))
    scores = np.asarray([e['score'] for e in events])
    best = np.asarray([e['best'] for e in events])
    np.testing.assert_allclose(best, np.minimum.accumulate(scores), rtol=1e-12, atol=1e-12)
    assert np.isclose(best[-1], final, rtol=1e-10, atol=1e-12)
    assert len(curve) == trace['steps']
    ends = {e['step']: e['best'] for e in events}
    np.testing.assert_allclose([ends[i] for i in range(1, len(curve) + 1)], curve, rtol=1e-12, atol=1e-12)
    first = next(e for e in events if np.isclose(e['best'], final, rtol=1e-10, atol=1e-12))
    assert first['evaluation'] == trace['first_final_evaluation']
    assert first['step'] == trace['first_final_step']


def run_case(case, pool, fingerprint):
    settings = Settings(sizes=[case['size']], data_seeds=[case['data_seed']],
                        algorithm_seeds=[case['algorithm_seed']] if case['algorithm_seed'] is not None else [42],
                        iterations=case['iterations'])
    settings.validate()
    budget = settings.budget(case['size'])
    common = dict(coverage_matrix=pool['coverage'], durations=pool['durations'],
                  dependency_matrix=pool['dependencies'], critical_feature_indices=pool['critical'],
                  min_coverage=settings.min_coverage, w_time=settings.w_time, w_count=settings.w_count,
                  w_redundancy=settings.w_redundancy, w_rare=settings.w_rare)
    trace = {}
    started = perf_counter()
    if case['method'] == 'IBMGO':
        selected, final, curve = binary_mountain_gazelle_optimizer(
            **common, **settings.options(case['size']), pop_size=12, max_iter=case['iterations'],
            random_seed=case['algorithm_seed'], trace=trace,
        )
        planned = budget['ibmgo']
    elif case['method'] == 'GA':
        selected, final, curve = baselines.genetic_algorithm_optimizer(
            **common, pop_size=12, max_generations=budget['generations'],
            random_seed=case['algorithm_seed'], trace=trace,
        )
        planned = budget['ga']
    else:
        # Greedy'nin gerçek puanlama çağrısını da say. Arama/onarım davranışına dokunma.
        with patch.object(baselines, 'calculate_fitness', wraps=baselines.calculate_fitness) as counter:
            selected, final = baselines.greedy_regression_selector(**common)
        assert counter.call_count == 1
        curve, planned = [], 1
        trace = dict(events=[dict(evaluation=1, step=0, phase='greedy', score=final, best=final)],
                     initial_fitness=final, final_fitness=final, steps=0, evaluations=1,
                     first_final_step=0, first_final_evaluation=1)
    elapsed = perf_counter() - started
    verify_trace(trace, curve, final, planned)
    direct_score = calculate_fitness(selected, **common)
    assert np.isclose(direct_score, final, rtol=1e-10, atol=1e-12)
    m = get_suite_metrics(selected, pool['coverage'], pool['durations'], pool['dependencies'], pool['critical'])
    valid = not m['critical_missing'] and not m['broken_dependencies'] and m['coverage_ratio'] >= .9 - 1e-12
    row = dict(kosum=case['case_id'], test_sayisi=case['size'], veri_seed=case['data_seed'],
               algoritma_seed=case['algorithm_seed'], yontem=case['method'], iterasyon=case['iterations'],
               baslangic_fitness=trace['initial_fitness'], fitness=final, secilen_test=m['selected_count'],
               tam_paket_sn=m['total_duration'], secili_paket_sn=m['selected_duration'],
               sure_tasarrufu_yuzde=100 * (1 - m['time_ratio']),
               kritik_kapsam_yuzde=100 * (1 - m['critical_missing'] / m['critical_total']),
               genel_kapsam_yuzde=100 * m['coverage_ratio'], on_kosul_ihlali=m['broken_dependencies'],
               arama_sn=elapsed, planlanan_degerlendirme=planned, gercek_degerlendirme=trace['evaluations'],
               ilk_en_iyi_adim=trace['first_final_step'], ilk_en_iyi_degerlendirme=trace['first_final_evaluation'],
               gecerli=bool(valid), girdi_kimligi=digest_pool(pool), deney_kimligi=fingerprint)
    return dict(case=case, record=row, trace=trace, curve=list(map(float, curve)),
                selection=list(map(int, selected)), settings=asdict(settings),
                resolved_options=settings.options(case['size']), budget=budget,
                fingerprint=fingerprint, completed_utc=timestamp())


def completed_records(folder, plan, fingerprint, inputs):
    expected = {case['case_id']: case for case in plan['cases']}
    found = {}
    for path in (folder / 'kosumlar').glob('*.json'):
        payload = json.loads(path.read_text(encoding='utf-8'))
        case_id = path.stem
        if case_id not in expected or payload['case'] != expected[case_id] or payload['fingerprint'] != fingerprint:
            raise ValueError(f'Farklı veya yinelenmiş deney kaydı. {path.name}')
        row = payload['record']
        if row['kosum'] != case_id or row['deney_kimligi'] != fingerprint:
            raise ValueError(f'Koşum kimliği tutarsız. {path.name}')
        input_key = f'{expected[case_id]["size"]}_{expected[case_id]["data_seed"]}'
        if row['girdi_kimligi'] != inputs[input_key]:
            raise ValueError(f'Koşum girdisi tutarsız. {path.name}')
        verify_trace(payload['trace'], payload['curve'], row['fitness'], row['planlanan_degerlendirme'])
        found[case_id] = row
    return found


def execute(kind, folder, smoke=False, limit=None):
    folder = Path(folder).resolve()
    if folder == (ROOT / 'sonuclar').resolve() or any(p.name.startswith('dayaniklilik_') for p in (folder, *folder.parents)):
        raise ValueError('Ek deney için ana arşivden ayrı bir klasör seçin.')
    folder.mkdir(parents=True, exist_ok=True)
    plan = experiment_plan(kind, smoke)
    started = perf_counter()
    with experiment_lock(folder):
        pools, manifest = prepare(folder, plan)
        found = completed_records(folder, plan, manifest['fingerprint'], manifest['identity']['inputs'])
        manifest.update(status='running', updated_utc=timestamp(), completed=len(found))
        atomic_json(folder / 'manifest.json', manifest)

        def save_index():
            frame = pd.DataFrame([found[c['case_id']] for c in plan['cases'] if c['case_id'] in found])
            if not frame.empty:
                atomic_csv(folder / 'ham_sonuclar.csv', frame)

        save_index()
        print(f'{kind} · {len(found)}/{len(plan["cases"])} kayıt hazır · {perf_counter()-started:.1f} sn', flush=True)
        newly_completed = 0
        try:
            for case in plan['cases']:
                if case['case_id'] in found:
                    continue
                if limit is not None and newly_completed >= limit:
                    break
                payload = run_case(case, pools[case['size'], case['data_seed']], manifest['fingerprint'])
                atomic_json(folder / 'kosumlar' / (case['case_id'] + '.json'), payload)
                found[case['case_id']] = payload['record']
                newly_completed += 1
                save_index()
                manifest.update(completed=len(found), updated_utc=timestamp())
                atomic_json(folder / 'manifest.json', manifest)
                print(f'{kind} · {len(found)}/{len(plan["cases"])} · {case["case_id"]} · '
                      f'puan {payload["record"]["fitness"]:.6f} · toplam {perf_counter()-started:.1f} sn', flush=True)
            manifest.update(status='complete' if len(found) == len(plan['cases']) else 'partial')
        except BaseException as exc:
            manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            raise
        finally:
            manifest.update(completed=len(found), updated_utc=timestamp(), session_elapsed_seconds=perf_counter()-started)
            atomic_json(folder / 'manifest.json', manifest)
            save_index()
    return pd.DataFrame([found[c['case_id']] for c in plan['cases'] if c['case_id'] in found])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['iteration', 'scale'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit', type=int, help='Bu oturumda en fazla kaç yeni koşum çalışsın.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('--limit negatif olamaz.')
    execute(args.experiment, args.output, args.smoke, args.limit)


if __name__ == '__main__':
    main()
