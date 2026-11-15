"""Kayıtlı ve yeni sonuçları ayrı tutan sunum paneli."""
from dataclasses import asdict
from html import escape
from io import BytesIO
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display

from kod.deneyler.sunum_deneyi import Settings, PresentationExperiment, selected_table
from kod.raporlama import sunum_sonuclari as report


class PresentationPanel:
    def __init__(self, experiment=None):
        import ipywidgets as w
        self.w = w
        self.experiment = experiment or PresentationExperiment()
        self.controls = {}
        self.updating = False
        self.inputs_dirty = False
        self.running = False
        self.live_result = None
        self.live_fingerprint = None
        self.archive_result = None
        self.status = w.HTML()
        self.progress = w.IntProgress(min=0, max=1, description='Koşum',
                                      layout=w.Layout(display='none', width='95%'))
        self.run_button = w.Button(description='Sonuçları güncelle', button_style='success',
                                   layout=w.Layout(width='190px'))
        self.archive_button = w.Button(description='Ana arşivi göster', layout=w.Layout(width='170px'))
        self.live_button = w.Button(description='Yeni sonucu göster', disabled=True,
                                    layout=w.Layout(width='170px'))
        self.mode = w.Dropdown(options=['Hızlı canlı demo', 'Tam dayanıklılık'], description='Hazır mod')
        self.budget_output = w.Output()
        self.data_output = w.Output()
        self._make_settings()
        self._make_editor()

        self.size_filter = w.Dropdown(description='Test büyüklüğü', style={'description_width': 'initial'})
        self.data_filter = w.Dropdown(description='Veri tohumu')
        self.scope = w.Dropdown(options=[('Havuz özeti', 'pool'), ('Seçili koşum', 'run')], description='Kapsam')
        self.run_filter = w.Dropdown(description='IBMGO koşumu', style={'description_width': 'initial'},
                                      layout=w.Layout(width='650px', max_width='100%'))
        self.method_filter = w.Dropdown(options=list(report.METHODS), value='IBMGO', description='Yöntem')
        self.trace_view = w.ToggleButtons(options=[('İterasyon', 'iteration'),
                                                  ('Eşit değerlendirme ölçeği', 'evaluations')],
                                          style={'button_width': '245px'})
        self.overview = w.VBox()
        self.results = w.VBox()
        self.gain_area = w.VBox()
        self.trace_area = w.VBox()
        self.detail = w.VBox([
            w.HBox([self.size_filter, self.data_filter, self.scope], layout=w.Layout(flex_flow='row wrap')),
            self.results,
        ])
        self.tabs = w.Tab(children=[self.overview, self.detail])
        self.tabs.set_title(0, 'Ana deneyin genel özeti')
        self.tabs.set_title(1, 'Tek havuzun ayrıntısı')
        self.result_view = w.VBox([
            w.HBox([self.archive_button, self.live_button, self.run_button],
                   layout=w.Layout(flex_flow='row wrap')),
            self.status, self.progress, self.tabs, self.settings_view, self.editor,
        ], layout=w.Layout(max_width='1140px', width='100%'))
        self.result_view.add_class('ibmgo-panel')
        self.size_filter.observe(self.filter_changed, names='value')
        self.data_filter.observe(self.filter_changed, names='value')
        self.scope.observe(self.render, names='value')
        self.run_filter.observe(self.render, names='value')
        self.method_filter.observe(self.render_gain, names='value')
        self.trace_view.observe(self.render_trace, names='value')
        self.run_button.on_click(self.run)
        self.archive_button.on_click(self.archive)
        self.live_button.on_click(self.show_live)
        self.changed()

    def _accordion(self, child, title):
        accordion = self.w.Accordion(children=[child], selected_index=None)
        accordion.set_title(0, title)
        return accordion

    def _make_settings(self):
        w = self.w
        settings = self.experiment.settings
        style = {'description_width': '170px'}
        for key, label in [('sizes', 'Test büyüklükleri'), ('data_seeds', 'Veri tohumları'),
                           ('algorithm_seeds', 'Algoritma tohumları')]:
            self.controls[key] = w.Text(value=','.join(map(str, getattr(settings, key))),
                                       description=label, style=style, layout=w.Layout(width='540px', max_width='100%'))
        for key, label in [('min_coverage', 'Asgari genel kapsam'), ('w_time', 'Süre ağırlığı'),
                           ('w_count', 'Test sayısı ağırlığı'), ('w_redundancy', 'Tekrar ağırlığı'),
                           ('w_rare', 'Nadir kayıp ağırlığı')]:
            self.controls[key] = w.FloatText(value=getattr(settings, key), description=label, style=style)
        for key, label in [('population', 'Popülasyon'), ('iterations', 'IBMGO iterasyonu'),
                           ('local_trials', 'Yerel aday / iterasyon')]:
            self.controls[key] = w.IntText(value=getattr(settings, key), description=label, style=style)
        self.controls['adaptive_policy'] = w.Checkbox(value=settings.adaptive_policy, description='Boyuta duyarlı politika')
        self.controls['advanced'] = w.Textarea(value=json.dumps(settings.advanced), description='İleri ayarlar JSON',
                                                style=style, layout=w.Layout(width='620px', max_width='100%'))
        for control in self.controls.values():
            control.observe(self.changed, names='value')
        self.mode.observe(self.preset, names='value')
        self.settings_view = self._accordion(
            w.VBox([self.mode, *self.controls.values(), self.budget_output]),
            'Teknik ayrıntılar · Yeni deney ayarları',
        )

    def _make_editor(self):
        w = self.w
        self.pool_choice = w.Dropdown(description='Veri kopyası', layout=w.Layout(width='470px'))
        self.test_index = w.BoundedIntText(value=0, min=0, max=49, description='Test indeksi')
        self.feature_index = w.BoundedIntText(value=0, min=0, max=45, description='Hedef indeksi')
        self.duration = w.FloatText(value=1, description='Süre (sn)')
        self.is_critical = w.Checkbox(description='Kritik hedef')
        self.covered = w.Checkbox(description='Bu test bu hedefi kapsar')
        self.apply_button = w.Button(description='Veri değişikliğini uygula', layout=w.Layout(width='230px'))
        self.reset_button = w.Button(description='Orijinal veriye dön')
        self.apply_button.on_click(self.apply_edit)
        self.reset_button.on_click(self.reset)
        for control in (self.pool_choice, self.test_index, self.feature_index):
            control.observe(self.show_data, names='value')
        for control in (self.duration, self.is_critical, self.covered):
            control.observe(self.edit_pending, names='value')
        self.editor = self._accordion(w.VBox([
            self.pool_choice, w.HBox([self.test_index, self.feature_index]),
            w.HBox([self.duration, self.is_critical, self.covered]),
            w.HBox([self.apply_button, self.reset_button]), self.data_output,
        ]), 'Teknik ayrıntılar · Veri kopyasını düzenle')

    def read_settings(self):
        values = {}
        for key, control in self.controls.items():
            value = control.value
            if key in ('sizes', 'data_seeds', 'algorithm_seeds'):
                try:
                    value = [int(v.strip()) for v in value.split(',') if v.strip()]
                except ValueError as exc:
                    raise ValueError('Boyut ve tohum alanlarında virgülle ayrılmış tam sayılar kullanın.') from exc
            if key == 'advanced':
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError('İleri ayarlar geçerli bir JSON nesnesi olmalı.') from exc
            values[key] = value
        return Settings(mode=self.mode.value, **values).validate()

    def preset(self, change=None):
        config = Settings.full() if self.mode.value == 'Tam dayanıklılık' else Settings()
        self.updating = True
        for key, control in self.controls.items():
            value = getattr(config, key)
            control.value = (','.join(map(str, value)) if isinstance(value, list)
                             else json.dumps(value) if key == 'advanced' else value)
        self.updating = False
        self.changed()

    def _dirty_message(self):
        self.status.value = report.block('<p class="ibmgo-warning">Sonuçlar önceki ayarlara ait. '
                                         'Uygulanan değişiklikleri hesaplamak için Sonuçları güncelle düğmesine basın.</p>')

    def changed(self, change=None):
        if self.updating:
            return
        self.inputs_dirty = True
        try:
            self.experiment.settings = self.read_settings()
            self.updating = True
            pairs = [(f'{n} test · Veri {seed}', (n, seed))
                     for n in self.experiment.settings.sizes for seed in self.experiment.settings.data_seeds]
            old = self.pool_choice.value
            self.pool_choice.options = pairs
            self.pool_choice.value = old if old in [p[1] for p in pairs] else pairs[0][1]
            self.updating = False
            with self.budget_output:
                clear_output(wait=True)
                budgets = [{'Test büyüklüğü': n, 'IBMGO · planlanan': self.experiment.settings.budget(n)['ibmgo'],
                            'GA · planlanan': self.experiment.settings.budget(n)['ga'],
                            'GA nesli': self.experiment.settings.budget(n)['generations']}
                           for n in self.experiment.settings.sizes]
                display(pd.DataFrame(budgets))
            self._dirty_message()
            self.run_button.disabled = False
            self.show_data()
        except (ValueError, TypeError) as exc:
            self.updating = False
            self.status.value = report.block('<p class="ibmgo-warning">Geçersiz ayar. ' + escape(str(exc)) + '</p>')
            self.run_button.disabled = True
        if hasattr(self, 'tabs'):
            self.render()

    def edit_pending(self, change=None):
        if not self.updating:
            self.inputs_dirty = True
            self.status.value = report.block('<p class="ibmgo-warning">Sonuçlar önceki ayarlara ait. '
                                             'Veri değişikliğini uygula, ardından Sonuçları güncelle düğmesine basın.</p>')

    def show_data(self, change=None):
        if self.updating or self.pool_choice.value is None:
            return
        n, seed = self.pool_choice.value
        pool = self.experiment.pool(n, seed)
        self.updating = True
        self.test_index.max = n - 1
        self.feature_index.max = pool['coverage'].shape[1] - 1
        i, j = self.test_index.value, self.feature_index.value
        self.duration.value = float(pool['durations'][i])
        self.is_critical.value = j in pool['critical']
        self.covered.value = bool(pool['coverage'][i, j])
        self.updating = False
        with self.data_output:
            clear_output(wait=True)
            case = pool['metadata']['test_cases'][i]
            print(f"{case['id']} · {pool['names'][i]}\nBeklenen sonuç · {case['expected_result']}")
            print('Hedef · ' + pool['features'][j])
            rows = list(range(max(0, i - 3), min(n, i + 5)))
            cols = list(range(max(0, j - 3), min(len(pool['features']), j + 5)))
            display(pd.DataFrame(pool['coverage'][np.ix_(rows, cols)], index=rows, columns=cols))
            print('İndeksler sıfırdan başlar. Değişiklikler seçilen havuzun sunum kopyasına uygulanır.')

    def apply_edit(self, _=None):
        n, seed = self.pool_choice.value
        pool = self.experiment.pool(n, seed)
        i, j = self.test_index.value, self.feature_index.value
        pool['durations'][i] = self.duration.value
        pool['coverage'][i, j] = int(self.covered.value)
        critical = set(pool['critical'].tolist())
        if self.is_critical.value:
            critical.add(j)
        else:
            critical.discard(j)
        pool['critical'] = np.asarray(sorted(critical), dtype=int)
        self.experiment.set_pool(n, seed, pool)
        self.changed()

    def reset(self, _=None):
        self.experiment.reset_pool(*self.pool_choice.value)
        self.changed()

    def archive(self, _=None):
        try:
            if self.archive_result is None:
                self.archive_result = self.experiment.load_archive()
            self.experiment.result = self.archive_result
            self.inputs_dirty = False
            self.update_filters()
            self.tabs.selected_index = 0
            self.status.value = report.block('<p class="ibmgo-note">Kayıtlı ana deney gösteriliyor. Yeni deney ayarları aşağıdaki Teknik ayrıntılar bölümünde.</p>')
        except Exception as exc:
            self.status.value = report.block('<p class="ibmgo-warning">Arşiv yüklenemedi. ' + escape(str(exc)) + '</p>')

    def show_live(self, _=None):
        if self.live_result is None:
            return
        self.experiment.result = self.live_result
        self.experiment.result_fingerprint = self.live_fingerprint
        self.inputs_dirty = self.experiment.stale
        self.update_filters()
        self.tabs.selected_index = 1
        if self.inputs_dirty:
            self._dirty_message()
        else:
            self.status.value = report.block('<p class="ibmgo-note">Yeni çalıştırmanın sonuçları gösteriliyor.</p>')

    def run(self, _=None):
        if self.running:
            return
        self.running = True
        buttons = [self.run_button, self.archive_button, self.live_button, self.mode,
                   self.apply_button, self.reset_button, self.pool_choice,
                   self.test_index, self.feature_index, self.duration,
                   self.is_critical, self.covered, *self.controls.values()]
        for control in buttons:
            control.disabled = True
        self.progress.layout.display = ''
        self.progress.value = 0
        try:
            self.experiment.settings = self.read_settings()
            total = len(self.experiment.settings.sizes) * len(self.experiment.settings.data_seeds) * (2 * len(self.experiment.settings.algorithm_seeds) + 1)
            self.progress.max = total
            self.status.value = report.block(f'<p>Veri ve kısıtlar doğrulanıyor · 0/{total} koşum.</p>')

            def progress(done, count, key):
                self.progress.max = count
                self.progress.value = done
                self.status.value = report.block(f'<p>Hesaplanıyor · {done}/{count} koşum tamamlandı.</p>')

            self.live_result = self.experiment.run(progress)
            self.live_fingerprint = self.experiment.result_fingerprint
            self.inputs_dirty = False
            self.update_filters()
            self.tabs.selected_index = 1
            self.status.value = report.block(f'<p class="ibmgo-note">Yeni çalıştırma tamamlandı · {total}/{total} koşum.</p>')
        except Exception as exc:
            self.status.value = report.block('<p class="ibmgo-warning">Çalıştırma tamamlanamadı. ' + escape(str(exc)) + '</p>')
            self.render()
        finally:
            self.running = False
            self.progress.layout.display = 'none'
            for control in buttons:
                control.disabled = False
            self.live_button.disabled = self.live_result is None

    def update_filters(self):
        result = self.experiment.result
        self.updating = True
        old = self.size_filter.value
        sizes = sorted(result['rows']['size'].unique().tolist())
        self.size_filter.options = [(f'{size} test', size) for size in sizes]
        self.size_filter.value = old if old in sizes else sizes[0]
        self.tabs.set_title(0, 'Ana deneyin genel özeti' if result['source'] == 'Kayıtlı deney' else 'Yeni deneyin genel özeti')
        self.updating = False
        self.filter_changed()

    def filter_changed(self, change=None):
        if self.updating or self.experiment.result is None:
            return
        result = self.experiment.result
        self.updating = True
        rows = result['rows']
        seeds = sorted(rows.loc[rows['size'] == self.size_filter.value, 'data_seed'].unique().tolist())
        old = self.data_filter.value
        had_choices = bool(self.data_filter.options)
        self.data_filter.options = [('Tümü', None)] + [(str(seed), seed) for seed in seeds]
        self.data_filter.value = old if old in seeds or (old is None and had_choices) else seeds[0]
        runs = [(key, run) for key, run in result['runs'].items()
                if run['method'] == 'IBMGO' and run['size'] == self.size_filter.value
                and (self.data_filter.value is None or run['data_seed'] == self.data_filter.value)]
        old = self.run_filter.value
        self.run_filter.options = [(f'{r["size"]} test · Veri {r["data_seed"]} · Başlangıç {r["algorithm_seed"]} · IBMGO', key)
                                   for key, r in runs]
        keys = [key for key, _ in runs]
        self.run_filter.value = old if old in keys else keys[0] if keys else None
        self.run_filter.layout.display = '' if runs else 'none'
        self.scope.disabled = not bool(runs)
        if not runs:
            self.scope.value = 'pool'
        self.updating = False
        self.render()

    def selected_rows(self):
        result = self.experiment.result
        rows = result['rows']
        rows = rows[rows['size'] == self.size_filter.value]
        if self.data_filter.value is not None:
            rows = rows[rows['data_seed'] == self.data_filter.value]
        if self.scope.value == 'run' and self.run_filter.value:
            run = result['runs'][self.run_filter.value]
            rows = rows[(rows['data_seed'] == run['data_seed']) &
                        ((rows['algorithm_seed'] == run['algorithm_seed']) | (rows['method'] == 'Greedy'))]
        return rows

    def _html(self, text):
        return self.w.HTML(report.block(text))

    def _figure(self, fig):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=125, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return self.w.Image(value=buffer.getvalue(), format='png',
                            layout=self.w.Layout(width='100%', max_width='1100px'))

    def _replace(self, container, children):
        # Yeniden çizilen büyük tabloları ve PNG'leri widget belleğinde biriktirme.
        def tree(widget):
            yield widget
            for child in getattr(widget, 'children', ()):
                yield from tree(child)

        previous = [widget for child in container.children for widget in tree(child)]
        container.children = children
        retained = {id(widget) for child in children for widget in tree(child)}
        for widget in reversed(previous):
            if id(widget) not in retained:
                widget.close()

    def render(self, change=None):
        if self.updating or self.experiment.result is None:
            return
        result = self.experiment.result
        self._replace(self.overview, (self.w.HTML(report.overview_html(result)),
                                  self._figure(report.overview_figure(result)),
                                  self._technical(result, result['rows'], 'Arşiv' if result['source'] == 'Kayıtlı deney' else 'Yeni deney')))
        rows = self.selected_rows()
        self._replace(self.results, (
            self._html('<h3>A. Deney özeti</h3><p class="ibmgo-summary">' + escape(report.experiment_summary(result, rows)) + '</p>'),
            self._html('<h3>B. Yöntem karşılaştırması</h3>'),
            self.w.HTML(report.comparison_html(result, rows)),
            self._figure(report.package_figure(rows)),
            self._html('<h3>C. Önce–sonra kazanımı</h3>'), self.method_filter, self.gain_area,
            self._html('<h3>D. İterasyon boyunca iyileşme</h3>'), self.run_filter,
            self.trace_view, self.trace_area,
            self._technical(result, rows, 'Seçili sonuç'),
        ))
        self.render_gain()
        self.render_trace()

    def render_gain(self, change=None):
        if self.updating or self.experiment.result is None:
            return
        rows = self.selected_rows()
        method = self.method_filter.value
        self._replace(self.gain_area, (
            self._figure(report.gain_figure(rows, method)),
            self._html('<p>' + escape(report.gain_note(rows, method)) + '</p>'),
        ))

    def render_trace(self, change=None):
        if self.updating or self.experiment.result is None:
            return
        result, key = self.experiment.result, self.run_filter.value
        fig = report.convergence_figure(result, key, self.trace_view.value)
        self.trace_view.layout.display = '' if fig is not None else 'none'
        if fig is None:
            self._replace(self.trace_area, (self._html('<p class="ibmgo-note">' + report.NO_TRACE + '</p>'),))
        else:
            self._replace(self.trace_area, (
                self._figure(fig),
                self._html('<p>' + escape(report.convergence_note(result, key)) + '</p>'),
            ))

    def _technical(self, result, rows, label):
        settings = asdict(Settings(**result['settings']))
        labels = {'mode': 'Deney modu', 'sizes': 'Test büyüklükleri', 'data_seeds': 'Veri tohumları',
                  'algorithm_seeds': 'Algoritma tohumları', 'min_coverage': 'Asgari genel kapsam',
                  'w_time': 'Süre ağırlığı', 'w_count': 'Test sayısı ağırlığı',
                  'w_redundancy': 'Tekrar ağırlığı', 'w_rare': 'Nadir kayıp ağırlığı',
                  'population': 'Popülasyon', 'iterations': 'IBMGO iterasyonu',
                  'local_trials': 'Yerel aday / iterasyon', 'adaptive_policy': 'Boyuta duyarlı politika',
                  'advanced': 'İleri ayarlar'}
        settings_html = pd.Series({labels[k]: str(v) for k, v in settings.items()}, name='Değer').to_frame().to_html(escape=True)
        content = '<details><summary>Bu sonucun ayarları</summary>' + settings_html + '</details>'
        content += '<details><summary>Kaynak kayıtları</summary><pre>' + escape('\n'.join(result['paths'])) + '</pre></details>'
        content += '<details><summary>Ham sonuçlar</summary>' + rows.to_html(index=False, escape=True) + '</details>'
        key = self.run_filter.value
        if key and key in result['runs']:
            run = result['runs'][key]
            trace = run['trace']
            content += '<details><summary>Gerçek değerlendirme izi</summary>'
            content += '<pre>' + escape(json.dumps({k: v for k, v in trace.items() if k != 'events'}, ensure_ascii=False, indent=2)) + '</pre>'
            content += pd.DataFrame(trace.get('events', [])).to_html(index=False) + '</details>'
            table = selected_table(result, key, limit=len(run['order']))
            table = table.rename(columns={'sira': 'Sıra', 'test': 'Test', 'senaryo': 'Senaryo', 'sure_sn': 'Süre (sn)', 'on_kosul': 'Ön koşul'})
            content += '<details><summary>Seçilen testler ve sıra</summary>' + table.to_html(index=False, escape=True) + '</details>'
        content = '<div style="max-height:520px;overflow:auto">' + content + '</div>'
        return self._accordion(self.w.HTML(report.block(content)), 'Teknik ayrıntılar · ' + label)
