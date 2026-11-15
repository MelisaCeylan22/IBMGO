"""Sunum girdileri, izleme ve kaynak ayrımı için regresyon kontrolleri."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from kod.algoritmalar import bmgo, baselines
from kod.algoritmalar.izleme import EvaluationTrace
from kod.deneyler.sunum_deneyi import (Settings, PresentationExperiment, make_pool,
                                       validate_pool, digest_pool, result_figures,
                                       convergence_figures, summarize)


class PresentationTests(unittest.TestCase):
    def test_settings_and_budget(self):
        self.assertEqual(Settings().budget(50),dict(ibmgo=1212,ga=1212,generations=100,difference=0))
        s=Settings(population=7,iterations=3,local_trials=2,advanced={'spiral_search_trials':3})
        b=s.validate().budget(50)
        self.assertLessEqual(abs(b['difference']),s.population/2)
        for settings in [Settings(sizes=[49]),Settings(w_count=-1),Settings(min_coverage=float('nan')),
                         Settings(advanced={'local_search_trials':4}),Settings(population=2.5),
                         Settings(algorithm_seeds=[42,42])]:
            with self.assertRaises(ValueError):settings.validate()

    def test_invalid_pool(self):
        p=make_pool(50,83)
        p['durations'][0]=np.inf
        with self.assertRaisesRegex(ValueError,'süre'):validate_pool(p,.9)
        p=make_pool(50,83);p['dependencies'][1,0]=1
        with self.assertRaisesRegex(ValueError,'Ön koşul'):validate_pool(p,.9)
        p=make_pool(50,83);p['coverage'][:,p['critical'][0]]=0
        with self.assertRaisesRegex(ValueError,'kritik'):validate_pool(p,.9)
        p=make_pool(50,83);p['critical']=np.array([],dtype=int);p['coverage'][:]=0
        with self.assertRaisesRegex(ValueError,'Genel kapsam'):validate_pool(p,.9)

    def test_trace_preserves_search_and_counts_every_call(self):
        p=make_pool(50,83)
        args=dict(coverage_matrix=p['coverage'],durations=p['durations'],dependency_matrix=p['dependencies'],critical_feature_indices=p['critical'],min_coverage=.9,pop_size=4,random_seed=43)
        for module,fn,opts in [(bmgo,bmgo.binary_mountain_gazelle_optimizer,dict(max_iter=3,local_search_trials=4,spiral_search_trials=2)),
                               (baselines,baselines.genetic_algorithm_optimizer,dict(max_generations=5))]:
            original=fn(**args,**opts);trace={}
            with patch.object(module,'calculate_fitness',wraps=module.calculate_fitness) as counter:
                actual=fn(**args,**opts,trace=trace)
            np.testing.assert_array_equal(original[0],actual[0])
            np.testing.assert_array_equal(original[2],actual[2])
            self.assertEqual(trace['evaluations'],counter.call_count)
            self.assertEqual(trace['initial_fitness'],trace['events'][3]['best'])
            self.assertTrue(np.all(np.diff([e['best'] for e in trace['events']])<=0))
            self.assertAlmostEqual(trace['events'][-1]['best'],actual[1])
            for step,value in enumerate(actual[2],1):
                self.assertAlmostEqual(value,[e for e in trace['events'] if e['step']==step][-1]['best'])
            if module==bmgo:
                self.assertEqual(trace['evaluations'],4+3*(16+4+2))
                self.assertEqual(sum(e['phase']=='local' for e in trace['events']),12)
                self.assertEqual(sum(e['phase']=='spiral' for e in trace['events']),6)

    def test_local_improvement_first_access_tolerance(self):
        target={};t=EvaluationTrace(target)
        t.record(.8,0,'initial');t.finish_initial(.8)
        t.record(.7,1,'mgo');t.record(.5,1,'local');t.record(.5-1e-13,2,'mgo')
        t.finish(.5-1e-13,2)
        self.assertEqual(target['first_final_evaluation'],3)
        self.assertEqual(target['last_improvement_phase'],'local')

    def test_archive_design_and_missing_trace(self):
        e=PresentationExperiment();r=e.load_archive()
        self.assertEqual(len(r['rows']),315)
        self.assertFalse(r['runs'])
        self.assertEqual(r['rows'].groupby('method').size().to_dict(),{'GA':150,'Greedy':15,'IBMGO':150})
        self.assertTrue(r['rows'].actual_evaluations.isna().all())
        self.assertEqual(len(summarize(r['rows'])),15)

    def test_changed_inputs_all_methods_and_persistence(self):
        with TemporaryDirectory() as folder:
            e=PresentationExperiment(Settings(population=4,iterations=2,local_trials=2),root=folder)
            original=e.pool(50,83);edited=e.pool(50,83)
            edited['durations'][0]+=15
            edited['coverage'][1,0]=1-edited['coverage'][1,0]
            # Kritik olmayan bir hedefin kritiklik işaretini değiştir.
            edited['critical']=np.unique(np.append(edited['critical'],0))
            e.set_pool(50,83,edited)
            observed=[]
            def observe(fn):
                def call(**kwargs):
                    observed.append((kwargs['coverage_matrix'].copy(),kwargs['durations'].copy(),kwargs['critical_feature_indices'].copy()))
                    return fn(**kwargs)
                return call
            with patch('kod.deneyler.sunum_deneyi.binary_mountain_gazelle_optimizer',side_effect=observe(bmgo.binary_mountain_gazelle_optimizer)),patch('kod.deneyler.sunum_deneyi.genetic_algorithm_optimizer',side_effect=observe(baselines.genetic_algorithm_optimizer)),patch('kod.deneyler.sunum_deneyi.greedy_regression_selector',side_effect=observe(baselines.greedy_regression_selector)):
                r=e.run()
            self.assertEqual(len(observed),3)
            for c,d,critical in observed:
                np.testing.assert_array_equal(c,edited['coverage']);np.testing.assert_array_equal(d,edited['durations']);np.testing.assert_array_equal(critical,edited['critical'])
            self.assertEqual(r['rows'].input_hash.nunique(),1)
            self.assertFalse(e.stale)
            e.settings.w_time=.5;self.assertTrue(e.stale)
            e.settings.w_time=.4;self.assertFalse(e.stale)
            e.reset_pool(50,83);self.assertTrue(e.stale)
            self.assertEqual(digest_pool(original),digest_pool(e.pool(50,83)))
            saved=Path(r['paths'][0]);manifest=json.loads((saved/'ayarlar.json').read_text(encoding='utf8'))
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(len(json.loads((saved/'izler_ve_secimler.json').read_text(encoding='utf8'))),3)
            fig=result_figures(r,50)
            self.assertEqual(len(fig.axes),2)
            self.assertAlmostEqual(fig.axes[0].patches[0].get_height(),r['rows'].iloc[0].secili_paket_sn / 60)
            key=next(k for k in r['runs'] if 'IBMGO' in k)
            fig2=convergence_figures(r,key)
            self.assertAlmostEqual(fig2.axes[0].lines[0].get_ydata()[0],r['runs'][key]['trace']['initial_fitness'])
            import matplotlib.pyplot as plt
            plt.close('all')

    def test_validation_before_any_method(self):
        e=PresentationExperiment();p=e.pool(50,83);p['coverage'][:]=0;e.set_pool(50,83,p)
        with patch('kod.deneyler.sunum_deneyi.binary_mountain_gazelle_optimizer') as fn:
            with self.assertRaises(ValueError):e.run()
            fn.assert_not_called()

    def test_filters_do_not_run_optimization_and_edit_marks_dirty(self):
        from kod.raporlama.sunum_paneli import PresentationPanel
        panel=PresentationPanel()
        self.assertEqual(panel.pool_choice.value,(50,83))
        # Görsel çizimi ayrı uçtan uca notebook kontrolünde yapılır.
        panel.render=lambda *a,**kw:None
        panel.archive()
        self.assertEqual(panel.size_filter.value,50)
        with patch.object(panel.experiment,'run') as run:
            panel.size_filter.value=100
            panel.method_filter.value='GA'
            panel.trace_view.value='evaluations'
            run.assert_not_called()
        panel.controls['w_time'].value=.6
        self.assertTrue(panel.inputs_dirty)
        self.assertIn('Sonuçlar önceki ayarlara ait',panel.status.value)


class ResultScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kod.raporlama import sunum_sonuclari
        cls.report = sunum_sonuclari
        cls.archive = PresentationExperiment().load_archive()

    def test_archive_uses_saved_ratios_and_has_no_trace_axes(self):
        rows = self.archive['rows'].query('size == 250')
        fig = self.report.gain_figure(rows, 'IBMGO')
        self.assertEqual(fig.axes[0].patches[0].get_width(), 100)
        self.assertAlmostEqual(fig.axes[0].patches[1].get_width(),
                               100 - rows.query("method == 'IBMGO'").tasarruf_yuzde.mean())
        self.assertIn('oranı', fig.axes[0].get_title())
        self.assertIsNone(self.report.convergence_figure(self.archive, None))
        self.assertIn('Mutlak dakika', self.report.gain_note(rows, 'IBMGO'))
        plt = __import__('matplotlib.pyplot', fromlist=['close'])
        plt.close('all')

    def test_compact_table_and_numeric_constraint_checks(self):
        rows = self.archive['rows'].query('size == 250').copy()
        html = self.report.comparison_html(self.archive, rows)
        self.assertEqual(html.count('Sağlandı'), 3)
        self.assertIn('Ortalama ±', html)
        self.assertIn('IBMGO 30 koşum', html)
        self.assertIn('Greedy 3 koşum', html)
        self.assertIn('Kayıt yok', html)
        rows.loc[rows.method == 'IBMGO', 'on_kosul_ihlali'] = 1
        rows.loc[rows.method == 'GA', 'genel_kapsam_yuzde'] = np.nan
        html = self.report.comparison_html(self.archive, rows)
        self.assertEqual(html.count('Sağlandı'), 1)
        self.assertEqual(html.count('Kontrol edilmeli'), 2)

    def test_highlight_tolerance_uses_visible_precision(self):
        mask = self.report._best_mask([.123441, .123449, .4])
        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertEqual(self.report._best_mask([.1, .1 + 1e-12, .1]).tolist(), [False] * 3)
        self.assertNotEqual(self.report._value(.00000012, seconds=True), '0')

    def test_overview_is_categorical_and_separates_sizes(self):
        html = self.report.overview_html(self.archive)
        self.assertIn('150 koşum (15 veri kümesi × 10 tekrar)', html)
        self.assertIn('15 koşum (15 veri kümesi × 1 tekrar)', html)
        fig = self.report.overview_figure(self.archive)
        self.assertEqual([t.get_text() for t in fig.axes[0].get_xticklabels()], ['50', '100', '250', '500', '1000'])
        self.assertEqual(fig.axes[0].get_ylim()[0], 0)
        __import__('matplotlib.pyplot', fromlist=['close']).close('all')

    def test_live_multiple_repeats_and_closed_details(self):
        from kod.raporlama.sunum_paneli import PresentationPanel
        rows = self.archive['rows'].query('size == 250 and data_seed == 83').copy()
        rows = rows[rows.algorithm_seed.isin([42, 43]) | (rows.method == 'Greedy')]
        result = dict(self.archive, rows=rows, source='Yeni çalıştırma')
        summary = self.report.experiment_summary(result, rows)
        self.assertIn('IBMGO 2 tekrar/veri', summary)
        self.assertIn('GA 2 tekrar/veri', summary)
        self.assertIn('Greedy 1 tekrar/veri', summary)
        html = self.report.comparison_html(result, rows)
        self.assertIn('Ortalama ±', html)
        panel = PresentationPanel()
        panel.archive()
        self.assertEqual(panel.data_filter.value, 83)
        self.assertIsNone(panel.settings_view.selected_index)
        self.assertIsNone(panel.editor.selected_index)
        self.assertIsNone(panel.results.children[-1].selected_index)
        previous_image = panel.gain_area.children[0]
        with patch.object(panel.experiment, 'run') as optimize:
            panel.method_filter.value = 'GA'
            panel.data_filter.value = 97
            panel.size_filter.value = 250
            optimize.assert_not_called()
        self.assertIsNone(previous_image.comm)
        self.assertIsNotNone(panel.method_filter.comm)
        self.assertEqual(len(panel.results.children), 12)

    def test_initial_plateau_and_first_access_come_from_real_trace(self):
        run = dict(size=250, data_seed=83, algorithm_seed=42, method='IBMGO',
                   curve=[.7, .5, .5], trace=dict(initial_fitness=.8, final_fitness=.5, steps=3))
        result = dict(runs={'run': run})
        self.assertIn('2. iterasyonda', self.report.convergence_note(result, 'run'))
        fig = self.report.convergence_figure(result, 'run')
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [.8, .7, .5, .5])
        np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets()[0], [2, .5])
        run['trace']['initial_fitness'] = .5
        run['curve'] = [.5, .5, .5]
        self.assertIn('başlangıçtan sonra daha iyi puan bulunmadı', self.report.convergence_note(result, 'run'))
        fig = self.report.convergence_figure(result, 'run')
        self.assertEqual(len(set(fig.axes[0].lines[0].get_ydata())), 1)
        __import__('matplotlib.pyplot', fromlist=['close']).close('all')

    def test_single_run_has_one_context_and_integer_count(self):
        rows = self.archive['rows'].query('size == 250 and data_seed == 83').copy()
        rows = rows[(rows.algorithm_seed == 42) | (rows.method == 'Greedy')]
        self.assertEqual(self.report.experiment_summary(self.archive, rows).count('Tek koşum'), 1)
        html = self.report.comparison_html(self.archive, rows)
        self.assertNotIn('±', html)
        self.assertNotIn('tek koşum', html)


if __name__=='__main__':unittest.main()
