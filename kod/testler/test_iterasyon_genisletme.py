"""Yeni iterasyon planının kapsamı ve önceki kayıtların korunması."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kod.deneyler import iterasyon_genisletme as extension
from kod.deneyler import ek_deneyler as base


class IterationExtensionTests(unittest.TestCase):
    def test_exactly_ninety_new_ibmgo_runs(self):
        plan = extension.extension_plan()
        self.assertEqual(plan['sizes'], [1000, 5000])
        self.assertEqual(len(plan['cases']), 90)
        self.assertEqual(len({c['case_id'] for c in plan['cases']}), 90)
        self.assertEqual({c['method'] for c in plan['cases']}, {'IBMGO'})
        self.assertEqual(plan['algorithm_seeds'], [42, 43, 44, 45, 46])
        self.assertEqual(plan['data_seeds'], [83, 97, 109])

    def test_resume_uses_same_original_scoring_function(self):
        with TemporaryDirectory() as folder:
            first = extension.execute_extension(folder, smoke=True, limit=1)
            self.assertEqual(len(first), 1)
            with patch.object(base, 'run_case', wraps=base.run_case) as run:
                result = extension.execute_extension(folder, smoke=True)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(len(result), 2)
            self.assertTrue(result.gecerli.all())
            with patch.object(base, 'run_case') as run:
                extension.execute_extension(folder, smoke=True)
                run.assert_not_called()

    def test_completed_experiments_cannot_be_outputs(self):
        for name in ('ek_iterasyon_2026-11-13', 'ek_olcek_2026-11-13'):
            with self.assertRaisesRegex(ValueError, 'Tamamlanan deney'):
                extension.execute_extension(base.ROOT / 'sonuclar' / name)

    def test_all_three_transitions_keep_fitness_and_time_savings_separate(self):
        import pandas as pd
        from kod.raporlama.iterasyon_birlesik import transition_pairs, summarize_transitions
        rows = pd.DataFrame([
            dict(test_sayisi=1000, veri_seed=83, algoritma_seed=42, iterasyon=20,
                 fitness=.4, sure_tasarrufu_yuzde=80., arama_sn=2.),
            dict(test_sayisi=1000, veri_seed=83, algoritma_seed=42, iterasyon=50,
                 fitness=.3, sure_tasarrufu_yuzde=79., arama_sn=5.),
            dict(test_sayisi=1000, veri_seed=83, algoritma_seed=42, iterasyon=100,
                 fitness=.35, sure_tasarrufu_yuzde=81., arama_sn=9.),
        ])
        paired = transition_pairs(rows)
        self.assertEqual(len(paired), 3)
        first = paired.iloc[0]
        self.assertAlmostEqual(first.fitness_farki, .1)
        self.assertAlmostEqual(first.tasarruf_farki_yuzde_puan, -1.)
        final = paired.iloc[2]
        self.assertEqual(final.sonuc, 'kötüleşti')
        self.assertAlmostEqual(final.ek_arama_sn, 4.)
        self.assertAlmostEqual(final.tasarruf_farki_yuzde_puan, 2.)
        summary = summarize_transitions(paired)
        self.assertEqual(summary.kotulesti.sum(), 1)

    def test_duplicate_or_unmatched_pairs_are_rejected(self):
        import pandas as pd
        from kod.raporlama.iterasyon_birlesik import transition_pairs
        row = dict(test_sayisi=1000, veri_seed=83, algoritma_seed=42, iterasyon=20,
                   fitness=.4, sure_tasarrufu_yuzde=80., arama_sn=2.)
        with self.assertRaisesRegex(ValueError, 'birden fazla'):
            transition_pairs(pd.DataFrame([row, row]))
        with self.assertRaisesRegex(ValueError, 'eksik'):
            transition_pairs(pd.DataFrame([row]))


if __name__ == '__main__':
    unittest.main()
