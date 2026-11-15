"""Ek deneylerin tasarımı, atomik kayıtları ve devam kimliği."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from kod.deneyler import ek_deneyler as experiments
from kod.deneyler.sunum_deneyi import Settings


class ExtraExperimentTests(unittest.TestCase):
    def test_exact_designs_and_budgets(self):
        iterations = experiments.experiment_plan('iteration')
        scale = experiments.experiment_plan('scale')
        self.assertEqual(len(iterations['cases']), 90)
        self.assertEqual(len(scale['cases']), 42)
        self.assertEqual(len({c['case_id'] for c in iterations['cases']}), 90)
        self.assertEqual(sum(c['method'] == 'Greedy' for c in scale['cases']), 6)
        for steps, budget in ((20, 1212), (50, 3012), (100, 6012)):
            self.assertEqual(Settings(iterations=steps).budget(250)['ibmgo'], budget)
        self.assertEqual(Settings().budget(5000)['generations'], 100)

    def test_resume_skips_completed_without_double_counting(self):
        with TemporaryDirectory() as folder:
            first = experiments.execute('iteration', folder, smoke=True, limit=1)
            self.assertEqual(len(first), 1)
            with patch.object(experiments, 'run_case', wraps=experiments.run_case) as run:
                result = experiments.execute('iteration', folder, smoke=True)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(len(result), 2)
            self.assertTrue(result['gecerli'].all())
            self.assertTrue((result.gercek_degerlendirme <= result.planlanan_degerlendirme).all())
            with patch.object(experiments, 'run_case') as run:
                experiments.execute('iteration', folder, smoke=True)
                run.assert_not_called()
            saved = list((Path(folder) / 'kosumlar').glob('*.json'))
            self.assertEqual(len(saved), 2)
            self.assertEqual(json.loads((Path(folder) / 'manifest.json').read_text())['status'], 'complete')

    def test_modified_saved_input_is_rejected(self):
        with TemporaryDirectory() as folder:
            experiments.execute('iteration', folder, smoke=True, limit=0)
            path = Path(folder) / 'girdiler/50_83.npz'
            with np.load(path) as saved:
                values = dict(saved)
            values['durations'][0] += 1
            np.savez_compressed(path, **values)
            with patch.object(experiments, 'run_case') as run:
                with self.assertRaisesRegex(ValueError, 'girdi değişmiş'):
                    experiments.execute('iteration', folder, smoke=True)
                run.assert_not_called()

    def test_different_plan_and_record_identity_are_rejected(self):
        with TemporaryDirectory() as folder:
            experiments.execute('iteration', folder, smoke=True, limit=1)
            with self.assertRaisesRegex(ValueError, 'Devam edilemiyor'):
                experiments.execute('iteration', folder, smoke=False, limit=0)
            path = next((Path(folder) / 'kosumlar').glob('*.json'))
            value = json.loads(path.read_text(encoding='utf-8'))
            value['record']['girdi_kimligi'] = 'different'
            experiments.atomic_json(path, value)
            with self.assertRaisesRegex(ValueError, 'girdisi tutarsız'):
                experiments.execute('iteration', folder, smoke=True)

    def test_scale_smoke_has_independent_greedy_and_real_traces(self):
        with TemporaryDirectory() as folder:
            result = experiments.execute('scale', folder, smoke=True)
            self.assertEqual(result.yontem.value_counts().to_dict(), {'IBMGO': 1, 'GA': 1, 'Greedy': 1})
            self.assertEqual(result.loc[result.yontem == 'Greedy', 'gercek_degerlendirme'].iloc[0], 1)
            self.assertTrue(result['gecerli'].all())

    def test_paired_report_preserves_loss_and_difference_direction(self):
        import pandas as pd
        from kod.raporlama.ek_deney_raporu import paired_iterations, paired_summary
        records = []
        for seed in (42, 43):
            for steps in (20, 50, 100):
                final = .4 if steps == 20 else (.3 if seed == 42 else .5)
                records.append(dict(test_sayisi=250, veri_seed=83, algoritma_seed=seed,
                                    iterasyon=steps, fitness=final, arama_sn=steps / 10))
        pairs = paired_iterations(pd.DataFrame(records))
        summary = paired_summary(pairs)
        self.assertEqual(len(pairs), 4)
        self.assertTrue((summary.iyilesti == 1).all())
        self.assertTrue((summary.kotulesti == 1).all())
        self.assertTrue(np.allclose(summary.fark_mean, 0))
        self.assertEqual(pairs.loc[pairs.algoritma_seed == 42, 'sonuc'].unique().tolist(), ['iyileşti'])
        self.assertEqual(pairs.loc[pairs.algoritma_seed == 43, 'sonuc'].unique().tolist(), ['kötüleşti'])
        self.assertAlmostEqual(summary.iloc[0].arama_carpani_mean, 2.5)


if __name__ == '__main__':
    unittest.main()
