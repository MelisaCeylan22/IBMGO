"""Sunum için sonuç tabloları ve tek soruyu yanıtlayan grafikler.

Bu modül yalnızca tamamlanmış sonuçları okur. Arama başlatmaz.
"""
from html import escape

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


METHODS = ("IBMGO", "GA", "Greedy")
COLORS = {"IBMGO": "#087F8C", "GA": "#8250B5", "Greedy": "#C75A15"}
INK = "#1A2A40"
MISSING = "Kayıt yok"
NO_TRACE = "Bu arşivde iterasyon geçmişi yok; yeni deney çalıştırarak görüntüleyebilirsiniz."
CSS = """
<style>
.ibmgo-screen {color:#1a2a40;font:14px/1.5 'Segoe UI',sans-serif;max-width:1120px}
.ibmgo-screen h3 {font-size:19px;margin:16px 0 8px;font-weight:650}
.ibmgo-screen p {margin:7px 0}
.ibmgo-summary {background:#edf5f7;border-left:4px solid #087f8c;padding:12px 16px;border-radius:5px;font-weight:600}
.ibmgo-table {border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px;margin:10px 0}
.ibmgo-table th {background:#f1f4f8;font-size:13px;font-weight:650;line-height:1.4}
.ibmgo-table th,.ibmgo-table td {padding:10px 9px;text-align:right;border-bottom:1px solid #e5eaf0;white-space:normal}
.ibmgo-table th:first-child,.ibmgo-table td:first-child {text-align:left}
.ibmgo-table .best {background:#eaf5f0;font-weight:650}
.ibmgo-table .ok {color:#22734e}
.ibmgo-table .fail {color:#9d341f}
.ibmgo-note {color:#53657b;font-size:12px}
.ibmgo-warning {background:#fff4da;color:#7b5111;padding:9px 13px;border-radius:5px}
.ibmgo-panel {font-family:'Segoe UI',sans-serif;color:#1a2a40}
.ibmgo-panel .lm-TabBar-tabLabel,.ibmgo-panel .p-TabBar-tabLabel {text-overflow:clip}
.ibmgo-panel [role="tab"],.jupyter-widgets .lm-TabBar-tab,.jupyter-widgets .p-TabBar-tab {
  min-width:240px!important;max-width:none!important;flex:0 0 240px!important
}
</style>
"""


def block(content):
    return CSS + '<div class="ibmgo-screen">' + content + '</div>'


def _table(headers, rows):
    return ('<table class="ibmgo-table"><thead><tr>'
            + ''.join('<th>' + h + '</th>' for h in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td' + attrs + '>' + value + '</td>'
                                      for value, attrs in row) + '</tr>' for row in rows)
            + '</tbody></table>')


def _cell(value, attrs=''):
    return escape(str(value)), attrs


def _value(value, decimals=2, seconds=False):
    if not np.isfinite(value):
        return MISSING
    if seconds:
        # Gerçek küçük süreleri sıfıra yuvarlama.
        return '0' if value == 0 else f'{value:.4g}'
    return f'{value:.{decimals}f}'


def _best_mask(means, decimals=4, maximize=False, seconds=False):
    values = np.asarray(means, dtype=float)
    finite = np.isfinite(values)
    mask = np.zeros(len(values), dtype=bool)
    if not finite.any():
        return mask
    best = np.max(values[finite]) if maximize else np.min(values[finite])
    shown = _value(best, decimals, seconds)
    for i, value in enumerate(values):
        mask[i] = bool(finite[i] and (np.isclose(value, best, rtol=1e-9, atol=1e-12)
                                    or _value(value, decimals, seconds) == shown))
    # Üç yöntem de eşitse bir kazanan vurgusu yapma.
    return mask if not mask.all() else np.zeros(len(values), dtype=bool)


def counts_note(rows):
    parts = []
    for method in METHODS:
        frame = rows[rows['method'] == method]
        counts = frame.groupby(['size', 'data_seed']).size()
        if counts.empty:
            continue
        repeats = str(int(counts.iloc[0])) if counts.nunique() == 1 else f'{counts.min()}–{counts.max()}'
        parts.append(f'{method} {len(frame)} koşum ({len(counts)} veri kümesi × {repeats} tekrar)')
    return ' · '.join(parts)


def experiment_summary(result, rows):
    sizes = sorted(rows['size'].unique())
    datasets = rows[['size', 'data_seed']].drop_duplicates()
    source = result['source']
    if len(sizes) == 1 and len(datasets) == 1 and all(
            len(rows[rows['method'] == m]) == 1 for m in METHODS):
        seeds = rows.loc[rows['method'] != 'Greedy', 'algorithm_seed'].dropna().unique()
        algorithm = f'Algoritma tohumu {int(seeds[0])}' if len(seeds) == 1 else 'Yöntem başına bir koşum'
        return (f'{sizes[0]} test · Veri tohumu {int(datasets.iloc[0]["data_seed"])} · '
                f'{algorithm} · Tek koşum · {source}')
    size_text = f'{sizes[0]} test' if len(sizes) == 1 else f'{len(sizes)} test büyüklüğü'
    dataset_text = (f'Veri tohumu {int(datasets.iloc[0]["data_seed"])}' if len(datasets) == 1
                    else f'{len(datasets)} veri kümesi')
    repetitions = []
    for method in METHODS:
        counts = rows[rows['method'] == method].groupby(['size', 'data_seed']).size()
        if not counts.empty:
            count = str(int(counts.iloc[0])) if counts.nunique() == 1 else f'{counts.min()}–{counts.max()}'
            repetitions.append(f'{method} {count} tekrar/veri')
    return f'{size_text} · {dataset_text} · ' + ' · '.join(repetitions) + f' · {source}'


def constraints(rows, minimum):
    critical = rows['kritik_kapsam_yuzde']
    general = rows['genel_kapsam_yuzde']
    dependencies = rows['on_kosul_ihlali']
    return (not rows.empty and critical.notna().all() and general.notna().all()
            and dependencies.notna().all() and np.isclose(critical, 100, rtol=0, atol=1e-8).all()
            and (general >= 100 * minimum - 1e-8).all() and (dependencies == 0).all())


def comparison_html(result, rows):
    minimum = result['settings']['min_coverage']
    groups = [rows[rows['method'] == method] for method in METHODS]
    specs = [('fitness', 1, 4, False), ('secilen_test', 1, 1, False),
             ('secili_paket_sn', 60, 2, False), ('tasarruf_yuzde', 1, 2, True),
             ('arama_sn', 1, 4, False)]
    highlights = []
    for metric, divisor, decimals, maximize in specs:
        highlights.append(_best_mask([g[metric].mean() / divisor for g in groups],
                                     decimals, maximize, seconds=metric == 'arama_sn'))
    formatted = []
    for i, (method, group) in enumerate(zip(METHODS, groups)):
        cells = [_cell(method)]
        for j, (metric, divisor, decimals, _) in enumerate(specs):
            values = group[metric].dropna() / divisor
            places = 0 if metric == 'secilen_test' and len(group) == 1 else decimals
            label = _value(values.mean(), places, metric == 'arama_sn')
            if metric == 'arama_sn' and values.mean() == 0 and result['source'] == 'Kayıtlı deney':
                label = '0 (kayıt)'
            if len(values) > 1:
                label += ' ± ' + _value(values.std(ddof=1), decimals, metric == 'arama_sn')
            cells.append(_cell(label, ' class="best"' if highlights[j][i] else ''))
        passed = constraints(group, minimum)
        cells.append(_cell('Sağlandı' if passed else 'Kontrol edilmeli',
                           ' class="ok"' if passed else ' class="fail"'))
        formatted.append(cells)
    content = _table(['Yöntem', 'Amaç puanı ↓', 'Seçilen<br>test sayısı ↓',
                      'Tahmini paket<br>süresi · dk ↓', 'Süre tasarrufu<br>· % ↑',
                      'Arama süresi<br>· sn ↓', 'Kısıtlar'], formatted)
    multiple = any(len(g) > 1 for g in groups)
    if multiple:
        note = 'Ortalama ± örnek standart sapma. ' + counts_note(rows) + '.'
        content += '<p class="ibmgo-note">' + escape(note) + '</p>'
    checks = []
    for method, group in zip(METHODS, groups):
        checks.append([_cell(method), _cell(_value(group['kritik_kapsam_yuzde'].min(), 2)),
                       _cell(_value(group['genel_kapsam_yuzde'].min(), 2)),
                       _cell(_value(group['on_kosul_ihlali'].max(), 0))])
    content += _table(['Doğrulama', 'Kritik kapsam · en az %', 'Genel kapsam · en az %',
                       'Ön koşul ihlali · en fazla'], checks)
    content += '<p class="ibmgo-note">' + escape(
        f'Kritik kapsam ≥ %100 · Genel kapsam ≥ %{100 * minimum:g} · Ön koşul ihlali = 0. '
        'Amaç puanı — düşük daha iyi.'
    ) + '</p>'
    if rows['secili_paket_sn'].isna().any():
        content += '<p class="ibmgo-note">Arşivde mutlak dakika bilgisi bulunmuyor. Süre grafikleri kaydedilen tasarruf oranını kullanır.</p>'
    if (rows['arama_sn'] == 0).any():
        content += '<p class="ibmgo-note">Arşivde 0 olarak kaydedilen arama süreleri kayıt hassasiyetiyle gösterilir.</p>'
    return block(content + '<p>' + escape(interpretation(rows)) + '</p>')


def interpretation(rows):
    means = rows.groupby('method')['fitness'].mean().reindex(METHODS)
    best = means.min()
    winners = [method for method, value in means.items()
               if np.isclose(value, best, rtol=1e-9, atol=1e-12) or f'{value:.4f}' == f'{best:.4f}']
    prefix = 'Bu koşumda' if all(len(rows[rows.method == m]) == 1 for m in METHODS) else 'Bu seçimin ortalamasında'
    first = (f'{prefix} en düşük amaç puanı ' + ', '.join(winners)
             + (' için eşit görünüyor.' if len(winners) > 1 else ' ile bulundu.'))
    fastest = rows.groupby('method')['arama_sn'].mean().idxmin()
    if fastest in winners:
        second = f'{fastest}, bu seçimde en düşük ortalama arama süresine de sahip.'
    else:
        second = f'{fastest} daha kısa arama süresiyle paket üretti; puan ve arama süresi ayrı ölçütlerdir.'
    return first + ' ' + second


def _axes_style(axes):
    for ax in np.asarray(axes, dtype=object).flat:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e4eaf1')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=11, colors=INK)
        ax.title.set_color(INK)
        ax.title.set_fontsize(14)
        ax.title.set_weight('semibold')


def package_figure(rows):
    """İki panelde paket süresi ve test sayısı."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout='constrained', facecolor='white')
    fig.suptitle('Hangi yöntem nasıl bir paket seçti?', fontsize=16, fontweight='semibold', color=INK)
    absolute = rows['secili_paket_sn'].notna().all()
    for ax, metric, divisor, title in [
        (axes[0], 'secili_paket_sn' if absolute else 'tasarruf_yuzde', 60 if absolute else 1,
         'Tahmini paket süresi · dakika' if absolute else 'Kalan süre · tam paket = %100'),
        (axes[1], 'secilen_test', 1, 'Seçilen test sayısı'),
    ]:
        values = []
        for method in METHODS:
            group = rows.loc[rows.method == method, metric] / divisor
            values.append(group.mean() if absolute or metric != 'tasarruf_yuzde' else 100 - group.mean())
        bars = ax.bar(METHODS, values, color=[COLORS[m] for m in METHODS], width=.56)
        single = metric == 'secilen_test' and all(len(rows[rows.method == m]) == 1 for m in METHODS)
        ax.bar_label(bars, labels=[f'{v:.0f}' if single else f'{v:.1f}' for v in values],
                     padding=6, fontsize=12, color=INK)
        ax.set_title(title, pad=14)
        ax.set_ylim(0, max(values) * 1.23 if max(values) > 0 else 1)
    _axes_style(axes)
    return fig


def gain_figure(rows, method):
    """Aynı sonuç alt kümesinde seçilen yöntemin önce–sonra görünümü."""
    frame = rows[rows.method == method]
    absolute = frame[['tam_paket_sn', 'secili_paket_sn']].notna().all().all()
    if absolute:
        full = frame['tam_paket_sn'].mean() / 60
        selected = frame['secili_paket_sn'].mean() / 60
        title, xlabel, unit = 'Bütün testleri çalıştırmaya göre ne kazandık?', 'Tahmini süre · dakika', ' dk'
    else:
        full, selected = 100., 100 - frame['tasarruf_yuzde'].mean()
        title, xlabel, unit = 'Tam pakete göre tahmini süre oranı', 'Tam paket = %100', '%'
    fig, ax = plt.subplots(figsize=(10.5, 3.0), layout='constrained', facecolor='white')
    bars = ax.barh(['Bütün test havuzu', 'Seçilen paket'], [full, selected],
                   color=['#cbd4df', COLORS[method]], height=.46)
    ax.bar_label(bars, labels=[f'{v:.1f}{unit}' for v in (full, selected)], padding=7, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(full, selected) * 1.19)
    ax.set(title=title, xlabel=xlabel)
    _axes_style([ax])
    ax.grid(False)
    return fig


def gain_note(rows, method):
    frame = rows[rows.method == method]
    count, total = frame['secilen_test'].mean(), frame['size'].mean()
    count_text = f'{int(count)}' if len(frame) == 1 else f'{count:.1f}'
    label = 'Ort. süre tasarrufu' if len(frame) > 1 else 'Süre tasarrufu'
    text = f'{method} · {label} %{frame["tasarruf_yuzde"].mean():.2f} · Seçilen/toplam test {count_text}/{total:g}'
    if frame['tam_paket_sn'].isna().any():
        text += ' · Mutlak dakika bilgisi bu arşivde bulunmuyor.'
    return text


def paired_run(result, key, method):
    chosen = result['runs'].get(key)
    if chosen is None:
        return None
    return next((r for r in result['runs'].values()
                 if r['size'] == chosen['size'] and r['data_seed'] == chosen['data_seed']
                 and r['algorithm_seed'] == chosen['algorithm_seed'] and r['method'] == method), None)


def convergence_figure(result, key, view='iteration'):
    run = paired_run(result, key, 'IBMGO')
    if run is None or not run.get('trace'):
        return None
    fig, ax = plt.subplots(figsize=(10.5, 4.0), layout='constrained', facecolor='white')
    if view == 'iteration':
        values = np.asarray([run['trace']['initial_fitness']] + run['curve'])
        ax.step(np.arange(len(values)), values, where='post', color=COLORS['IBMGO'], lw=2.2, label='IBMGO')
        final = run['trace']['final_fitness']
        first = int(np.flatnonzero(np.isclose(values, final, rtol=1e-10, atol=1e-12))[0])
        ax.scatter([first], [values[first]], color=COLORS['IBMGO'], edgecolor='white', s=65, zorder=4)
        ax.axvline(first, color=COLORS['IBMGO'], ls=':', alpha=.6)
        ax.annotate(f'İlk erişim · {first}. iterasyon', (first, values[first]),
                    xytext=(8, 15), textcoords='offset points', fontsize=11,
                    ha='right' if first > len(values) / 2 else 'left')
        ax.set(xlabel='IBMGO iterasyonu · 0 = başlangıç', title='Arama ilerledikçe çözüm iyileşti mi?')
        ax.set_xlim(0, max(1, len(values) - 1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    elif view == 'evaluations':
        for method in ('IBMGO', 'GA'):
            other = paired_run(result, key, method)
            if other is None or not other.get('trace', {}).get('events'):
                continue
            events = other['trace']['events']
            ax.step([e['evaluation'] for e in events], [e['best'] for e in events],
                    where='post', color=COLORS[method], lw=2, label=method)
        ax.set(xlabel='Kümülatif gerçek fitness değerlendirmesi',
               title='Aynı değerlendirme ölçeğinde aramanın ilerleyişi')
    else:
        plt.close(fig)
        raise ValueError('Yakınsama görünümü geçersiz.')
    ax.set_ylabel('Amaç puanı — düşük daha iyi')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.margins(y=.2)
    ax.legend(frameon=False)
    _axes_style([ax])
    return fig


def convergence_note(result, key):
    run = paired_run(result, key, 'IBMGO')
    if run is None or not run.get('trace'):
        return NO_TRACE
    trace = run['trace']
    values = np.asarray([trace['initial_fitness']] + run['curve'])
    first = int(np.flatnonzero(np.isclose(values, trace['final_fitness'], rtol=1e-10, atol=1e-12))[0])
    if first == 0:
        return 'Bu koşumda başlangıçtan sonra daha iyi puan bulunmadı.'
    return (f'{trace["steps"]} iterasyon çalıştırıldı. Bu koşumun sonunda en iyi kalan puana '
            f'ilk kez {first}. iterasyonda ulaşıldı.')


def overview_html(result):
    rows = result['rows']
    data = rows.groupby(['size', 'method'])['fitness'].agg(['mean', 'std', 'size'])
    records = []
    for size in sorted(rows['size'].unique()):
        means = [data.loc[(size, m), 'mean'] for m in METHODS]
        best = _best_mask(means)
        record = [_cell(f'{size} test')]
        for i, method in enumerate(METHODS):
            r = data.loc[(size, method)]
            text = f'{r["mean"]:.4f}'
            if r['size'] > 1:
                text += f' ± {r["std"]:.4f}'
            record.append(_cell(text, ' class="best"' if best[i] else ''))
        records.append(record)
    content = '<p class="ibmgo-summary">' + escape(experiment_summary(result, rows)) + '</p>'
    content += '<h3>Amaç puanı — düşük daha iyi</h3>'
    content += _table(['Test büyüklüğü', *METHODS], records)
    content += '<p class="ibmgo-note">' + escape(counts_note(rows)) + '.</p>'
    content += ('<p class="ibmgo-note">Değerler ortalama ± örnek standart sapmadır. '
                'Veri ve algoritma tekrarları birlikte özetlenir. Puanlar her test büyüklüğü içinde karşılaştırılır.</p>')
    return block(content)


def overview_figure(result):
    rows = result['rows']
    sizes = sorted(rows['size'].unique())
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(11, 4.3), layout='constrained', facecolor='white')
    for i, method in enumerate(METHODS):
        data = rows[rows.method == method].groupby('size')['tasarruf_yuzde']
        means = data.mean().reindex(sizes)
        stds = data.std().reindex(sizes).fillna(0)
        ax.bar(x + (i - 1) * .24, means, width=.23, color=COLORS[method], label=method,
               yerr=stds, capsize=3, error_kw={'linewidth': 1, 'alpha': .7})
    ax.set(xticks=x, xticklabels=[str(n) for n in sizes], ylim=(0, 105),
           title='Test büyüklüğüne göre süre tasarrufu', xlabel='Test büyüklüğü',
           ylabel='Tam pakete göre tahmini süre tasarrufu (%)')
    ax.legend(frameon=False, ncol=3, loc='upper left')
    _axes_style([ax])
    return fig


def display_result(result, size=None, method='IBMGO', key=None):
    """Widget olmadan da aynı dört bölümü gerçek PNG çıktılarıyla göster."""
    from io import BytesIO
    from IPython.display import HTML, Image, display

    def show(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=125, bbox_inches='tight')
        plt.close(fig)
        display(Image(data=buffer.getvalue()))

    rows = result['rows']
    if size is None:
        display(HTML(overview_html(result)))
        show(overview_figure(result))
    else:
        rows = rows[rows['size'] == size]
        display(HTML(block('<h3>A. Deney özeti</h3><p class="ibmgo-summary">'
                           + escape(experiment_summary(result, rows)) + '</p>')))
        display(HTML(block('<h3>B. Yöntem karşılaştırması</h3>')))
        display(HTML(comparison_html(result, rows)))
        show(package_figure(rows))
        display(HTML(block('<h3>C. Önce–sonra kazanımı</h3>')))
        show(gain_figure(rows, method))
        display(HTML(block('<p>' + escape(gain_note(rows, method)) + '</p>')))
        display(HTML(block('<h3>D. İterasyon boyunca iyileşme</h3>')))
        fig = convergence_figure(result, key)
        if fig is not None:
            show(fig)
        display(HTML(block('<p>' + escape(convergence_note(result, key)) + '</p>')))
    details = ('<details><summary>Teknik ayrıntılar</summary><pre>'
               + escape(str(result['settings'])) + '\n' + escape('\n'.join(result['paths']))
               + '</pre>' + rows.to_html(index=False, escape=True) + '</details>')
    display(HTML(block(details)))
