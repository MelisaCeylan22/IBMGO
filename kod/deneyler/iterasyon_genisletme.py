"""Önceki hesaplama koduyla 1000 ve 5000 testte 90 yeni IBMGO koşumu.

python -B -m kod.deneyler.iterasyon_genisletme --output sonuclar/ek_iterasyon_1000_5000_2026-11-13
"""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import pandas as pd

from kod.deneyler import ek_deneyler as base


def extension_plan(smoke=False):
    plan = base.experiment_plan('iteration', smoke)
    if not smoke:
        mapping = {250: 1000, 500: 5000}
        plan['sizes'] = [1000, 5000]
        for case in plan['cases']:
            case['size'] = mapping[case['size']]
            case['case_id'] = (f'{case["size"]}_{case["data_seed"]}_IBMGO_'
                               f'{case["algorithm_seed"]}_{case["iterations"]}')
    plan['extension'] = '1000_5000'
    plan['extension_code'] = {
        str(Path(__file__).relative_to(base.ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    # Eski sonuçlara yazmadan, devam kimliğine onların içerik özetlerini de kat.
    plan['preserved_experiments'] = {
        str(p.relative_to(base.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ('ek_iterasyon_2026-11-13', 'ek_olcek_2026-11-13')
        for p in sorted((base.ROOT / 'sonuclar' / name).rglob('*')) if p.is_file()
    }
    return plan


def execute_extension(folder, smoke=False, limit=None):
    folder = Path(folder).resolve()
    protected = [base.ROOT / 'sonuclar' / name for name in (
        'ek_iterasyon_2026-11-13', 'ek_olcek_2026-11-13')]
    if folder == (base.ROOT / 'sonuclar').resolve() or any(
            p.name.startswith('dayaniklilik_') for p in (folder, *folder.parents)):
        raise ValueError('Yeni deney için ayrı bir klasör seçin.')
    if any(folder == p.resolve() or p.resolve() in folder.parents for p in protected):
        raise ValueError('Tamamlanan deney klasörleri yeni kayıtlar için kullanılamaz.')
    folder.mkdir(parents=True, exist_ok=True)
    plan = extension_plan(smoke)
    started = perf_counter()
    with base.experiment_lock(folder):
        pools, manifest = base.prepare(folder, plan)
        found = base.completed_records(folder, plan, manifest['fingerprint'], manifest['identity']['inputs'])
        manifest.update(status='running', updated_utc=base.timestamp(), completed=len(found))
        base.atomic_json(folder / 'manifest.json', manifest)

        def save_index():
            frame = pd.DataFrame([found[c['case_id']] for c in plan['cases'] if c['case_id'] in found])
            if not frame.empty:
                base.atomic_csv(folder / 'ham_sonuclar.csv', frame)

        save_index()
        print(f'Ek iterasyon · {len(found)}/{len(plan["cases"])} kayıt hazır · '
              f'{perf_counter() - started:.1f} sn', flush=True)
        newly_completed = 0
        try:
            for case in plan['cases']:
                if case['case_id'] in found:
                    continue
                if limit is not None and newly_completed >= limit:
                    break
                # Puanlama, onarım, politika ve gerçek iz önceki çalıştırıcıdan gelir.
                payload = base.run_case(case, pools[case['size'], case['data_seed']], manifest['fingerprint'])
                base.atomic_json(folder / 'kosumlar' / (case['case_id'] + '.json'), payload)
                found[case['case_id']] = payload['record']
                newly_completed += 1
                save_index()
                manifest.update(completed=len(found), updated_utc=base.timestamp())
                base.atomic_json(folder / 'manifest.json', manifest)
                print(f'Ek iterasyon · {len(found)}/{len(plan["cases"])} · {case["size"]} test · '
                      f'veri {case["data_seed"]} · başlangıç {case["algorithm_seed"]} · '
                      f'{case["iterations"]} iterasyon · puan {payload["record"]["fitness"]:.6f} · '
                      f'geçen {perf_counter() - started:.1f} sn', flush=True)
            manifest.update(status='complete' if len(found) == len(plan['cases']) else 'partial')
        except BaseException as exc:
            manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
            raise
        finally:
            manifest.update(completed=len(found), updated_utc=base.timestamp(),
                            session_elapsed_seconds=perf_counter() - started)
            base.atomic_json(folder / 'manifest.json', manifest)
            save_index()
    return pd.DataFrame([found[c['case_id']] for c in plan['cases'] if c['case_id'] in found])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit', type=int, help='Bu oturumda en fazla yeni koşum sayısı.')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error('--limit negatif olamaz.')
    execute_extension(args.output, args.smoke, args.limit)


if __name__ == '__main__':
    main()
