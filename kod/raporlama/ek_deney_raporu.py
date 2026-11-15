"""Tamamlanmış ek deneylerden CSV ve 16:9 sunum görselleri üretir."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from kod.deneyler.ek_deneyler import ROOT, atomic_csv


INK = '#172C43'
GRAY = '#56697F'
ITER_COLORS = {20: '#087F8C', 50: '#8250B5', 100: '#C75A15'}
METHOD_COLORS = {'IBMGO': '#087F8C', 'GA': '#8250B5', 'Greedy': '#C75A15'}
SCHEDULE_NOTE = ('Toplam iterasyon, katsayı ve bit değişim takvimini de değiştirir. '
                 'Bu deney farklı bütçe ve kontrol takvimlerini karşılaştırır.')


def load_records(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['completed'] != manifest['total']:
        raise ValueError(f'Deney henüz tamamlanmadı. {manifest["completed"]}/{manifest["total"]}')
    rows = pd.read_csv(folder / 'ham_sonuclar.csv')
    expected = {c['case_id'] for c in manifest['identity']['plan']['cases']}
    if set(rows['kosum']) != expected or len(rows) != len(expected):
        raise ValueError('Ham kayıtlar deney planıyla uyuşmuyor.')
    if set(rows['deney_kimligi']) != {manifest['fingerprint']}:
        raise ValueError('Sonuçların deney kimliği uyuşmuyor.')
    return rows, manifest


def aggregate(rows, groups):
    metrics = ['fitness', 'secilen_test', 'tam_paket_sn', 'secili_paket_sn', 'sure_tasarrufu_yuzde',
               'arama_sn', 'gercek_degerlendirme', 'kritik_kapsam_yuzde', 'genel_kapsam_yuzde', 'on_kosul_ihlali']
    grouped = rows.groupby(groups)
    values = grouped[metrics].agg(['mean', 'std', 'min', 'max'])
    values.columns = ['_'.join(c) for c in values.columns]
    counts = grouped.agg(kosum_sayisi=('kosum', 'size'), veri_sayisi=('veri_seed', 'nunique'),
                         algoritma_tohumu_sayisi=('algoritma_seed', 'nunique'), gecerli_oran=('gecerli', 'mean'))
    return counts.join(values).reset_index()


def paired_iterations(rows):
    reference = rows[rows.iterasyon == 20].set_index(['test_sayisi', 'veri_seed', 'algoritma_seed'])
    records = []
    for target in (50, 100):
        other = rows[rows.iterasyon == target].set_index(reference.index.names)
        if set(reference.index) != set(other.index):
            raise ValueError('İterasyon ayarlarının eşleşen tohumları eksik.')
        for key, base in reference.iterrows():
            candidate = other.loc[key]
            delta = float(base.fitness - candidate.fitness)
            equal = bool(np.isclose(base.fitness, candidate.fitness, rtol=1e-10, atol=1e-12))
            outcome = 'eşit' if equal else 'iyileşti' if delta > 0 else 'kötüleşti'
            records.append(dict(test_sayisi=key[0], veri_seed=key[1], algoritma_seed=key[2],
                                hedef_iterasyon=target, fitness_20=base.fitness, fitness_yeni=candidate.fitness,
                                fark_20_eksi_yeni=delta, sonuc=outcome,
                                arama_20_sn=base.arama_sn, arama_yeni_sn=candidate.arama_sn,
                                ek_arama_sn=candidate.arama_sn - base.arama_sn,
                                arama_suresi_carpani=candidate.arama_sn / base.arama_sn))
    return pd.DataFrame(records)


def paired_summary(pairs):
    records = []
    for (size, target), group in pairs.groupby(['test_sayisi', 'hedef_iterasyon']):
        records.append(dict(test_sayisi=size, hedef_iterasyon=target, cift=len(group),
                            fark_mean=group.fark_20_eksi_yeni.mean(), fark_std=group.fark_20_eksi_yeni.std(),
                            fark_medyan=group.fark_20_eksi_yeni.median(),
                            iyilesti=int((group.sonuc == 'iyileşti').sum()),
                            esit=int((group.sonuc == 'eşit').sum()), kotulesti=int((group.sonuc == 'kötüleşti').sum()),
                            ek_arama_mean=group.ek_arama_sn.mean(), ek_arama_std=group.ek_arama_sn.std(),
                            arama_carpani_mean=group.arama_suresi_carpani.mean()))
    return pd.DataFrame(records)


def _export(fig, folder, name):
    target = Path(folder) / 'sunum'
    target.mkdir(exist_ok=True)
    fig.canvas.draw()
    # Metin kutuları slayt sınırları içinde kalmalı.
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
    for text in texts:
        if not text.get_text():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
            raise ValueError(f'Slayt metni sınır dışına taşıyor. {text.get_text()}')
    fig.savefig(target / name, dpi=240, facecolor='white')
    plt.close(fig)


def _slide(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.05, .93, title, fontsize=28, fontweight='bold', color=INK)
    fig.text(.05, .875, subtitle, fontsize=17, color=GRAY)
    return fig


def _style(axes):
    for ax in np.asarray(axes, dtype=object).flat:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E1E8F0')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=16, colors=INK)
        ax.title.set(fontsize=21, fontweight='semibold', color=INK)
        ax.xaxis.label.set(fontsize=17, color=INK)
        ax.yaxis.label.set(fontsize=17, color=INK)


def table_slide(folder, name, title, subtitle, headers, body, widths, footer):
    fig = _slide(title, subtitle)
    ax = fig.add_axes([.045, .24, .91, .56])
    ax.axis('off')
    table = ax.table(cellText=body, colLabels=headers, colWidths=widths, cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#DFE6EF')
        cell.set_linewidth(.65)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.set_facecolor('#EAF1F6')
            cell.get_text().set_fontweight('semibold')
            cell.get_text().set_fontsize(15)
        else:
            cell.set_facecolor('#F8FAFC' if r % 2 == 0 else 'white')
    fig.text(.05, .15, footer, fontsize=16, color=GRAY, va='top', linespacing=1.6)
    _export(fig, folder, name)


def _pm(mean, std, digits=2):
    return f'{mean:.{digits}f} ± {std:.{digits}f}'


def _convergence(folder, rows, seed):
    fig = _slide(f'Arama bütçesi boyunca amaç puanı · Veri tohumu {seed}',
                 'Aynı veri üzerinde 5 algoritma başlangıcı · Çizgi ortalama, bant minimum–maksimum')
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.075, right=.97, bottom=.27, top=.77, wspace=.25)
    for ax, size in zip(axes, sorted(rows.test_sayisi.unique())):
        for steps, group in rows[(rows.test_sayisi == size) & (rows.veri_seed == seed)].groupby('iterasyon'):
            traces = []
            for case in group.kosum:
                payload = json.loads((Path(folder) / 'kosumlar' / f'{case}.json').read_text(encoding='utf-8'))
                traces.append(np.array([e['best'] for e in payload['trace']['events']]))
            common = min(map(len, traces))
            values = np.stack([t[:common] for t in traces])
            x = np.arange(1, common + 1)
            color = ITER_COLORS[int(steps)]
            ax.step(x, values.mean(axis=0), where='post', color=color, lw=2.3, label=f'{steps} iterasyon')
            ax.fill_between(x, values.min(axis=0), values.max(axis=0), step='post', color=color, alpha=.15)
            for trace in traces:
                if len(trace) > common:
                    ax.step(np.arange(common, len(trace) + 1), trace[common - 1:], where='post', color=color, alpha=.3)
        ax.set(title=f'{size} test', xlabel='Gerçek kümülatif\nfitness değerlendirmesi', ylabel='Amaç puanı · düşük daha iyi')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.legend(fontsize=15, frameon=False)
    _style(axes)
    fig.text(.05, .13, 'Bant bir güven aralığı değildir. Biten koşumun eğrisi ileri taşınmaz. Yumuşatma uygulanmaz.',
             fontsize=16, color=GRAY)
    fig.text(.05, .075, '20, 50 ve 100 iterasyon farklı toplam bütçe ve kontrol takvimleridir.', fontsize=16, color=GRAY)
    _export(fig, folder, f'02_yakinsama_veri{seed}.png')


def generate_iteration(folder):
    rows, manifest = load_records(folder)
    summary = aggregate(rows, ['test_sayisi', 'iterasyon'])
    pairs = paired_iterations(rows)
    paired = paired_summary(pairs)
    atomic_csv(Path(folder) / 'iterasyon_ozet.csv', summary)
    atomic_csv(Path(folder) / 'eslesmis_farklar.csv', pairs)
    atomic_csv(Path(folder) / 'eslesmis_ozet.csv', paired)
    body = []
    for _, r in summary.iterrows():
        evaluations = (f'{r.gercek_degerlendirme_mean:.0f}' if r.gercek_degerlendirme_std < 1e-10
                       else _pm(r.gercek_degerlendirme_mean, r.gercek_degerlendirme_std, 1))
        body.append([int(r.test_sayisi), int(r.iterasyon), int(r.kosum_sayisi), _pm(r.fitness_mean, r.fitness_std, 4),
                     f'{r.secilen_test_mean:.1f}', f'{r.sure_tasarrufu_yuzde_mean:.2f}',
                     f'{r.arama_sn_mean:.2f}', evaluations, f'%{100 * r.gecerli_oran:.0f}'])
    headers = ['Test', 'İterasyon', 'Koşum', 'Amaç puanı ↓\nort. ± std', 'Seçilen\ntest · ort.',
               'Tasarruf\n% · ort.', 'Arama\nsn · ort.', 'Gerçek\ndeğerl. · ort.', 'Geçerli\npaket']
    table_slide(folder, '01_iterasyon_ozet.png', 'Toplam arama bütçesi neyi değiştirdi?',
                'Her satır 3 veri tohumu × 5 algoritma başlangıcı = 15 koşum', headers, body,
                [.06, .075, .065, .18, .11, .11, .11, .17, .12],
                'Standart sapma örnek standart sapmasıdır. Aynı test büyüklüğü içinde karşılaştırın.\n'
                'Tasarruf sentetik test sürelerinden hesaplanır. Arama süresi bu ortamda, iz kaydı açıkken ölçülmüştür.')
    for seed in sorted(rows.veri_seed.unique()):
        _convergence(folder, rows, int(seed))
    body = [[int(r.test_sayisi), f'20 → {int(r.hedef_iterasyon)}', int(r.cift),
             _pm(r.fark_mean, r.fark_std, 5), f'{int(r.iyilesti)} / {int(r.esit)} / {int(r.kotulesti)}',
             _pm(r.ek_arama_mean, r.ek_arama_std, 2), f'{r.arama_carpani_mean:.2f}×'] for _, r in paired.iterrows()]
    table_slide(folder, '03_ek_fayda_tablosu.png', 'Ek aramanın faydası ve süre maliyeti',
                'Aynı veri ve algoritma tohumları eşleştirildi · Her geçişte 15 çift',
                ['Test', 'Geçiş', 'Çift', 'Δ amaç puanı\nort. ± std', 'İyileşti / eşit\n/ kötüleşti',
                 'Ek arama · sn\nort. ± std', 'Süre\nçarpanı · ort.'], body,
                [.07, .11, .07, .23, .18, .23, .11],
                'Δ = 20 iterasyon puanı − yeni puan. Pozitif değer daha düşük amaç puanını gösterir.\n'
                'Eşitlik toleransı rtol=1e−10, atol=1e−12. Sayısal farklar istatistiksel üstünlük kanıtı değildir.')
    fig = _slide('Eşleşen koşumlarda ek bütçenin puan farkı', 'Her nokta gerçek bir tohum çifti · Elmaslar ortalama')
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.2, wspace=.22)
    for ax, size in zip(axes, sorted(rows.test_sayisi.unique())):
        for position, steps in enumerate((50, 100)):
            values = pairs[(pairs.test_sayisi == size) & (pairs.hedef_iterasyon == steps)].fark_20_eksi_yeni
            offsets = np.linspace(-.11, .11, len(values))
            ax.scatter(position + offsets, values, color=ITER_COLORS[steps], alpha=.65, s=40)
            ax.scatter([position], [values.mean()], color=INK, marker='D', s=85, edgecolor='white', zorder=4)
        ax.axhline(0, color=GRAY, ls='--', lw=1.2)
        ax.set(title=f'{size} test', xticks=[0, 1], xticklabels=['20 → 50', '20 → 100'],
               xlabel='Toplam iterasyon geçişi', ylabel='Δ = F20 − Fyeni · pozitif daha iyi')
    _style(axes)
    fig.text(.05, .1, 'Başlangıçlar eşleştirilmiştir. Kontrol takvimleri toplam iterasyonla birlikte değişir.', fontsize=17, color=GRAY)
    _export(fig, folder, '04_eslesmis_farklar.png')
    schedules = []
    for (size, seed, start), group in rows.groupby(['test_sayisi', 'veri_seed', 'algoritma_seed']):
        cases = {int(r.iterasyon): json.loads((Path(folder) / 'kosumlar' / f'{r.kosum}.json').read_text(encoding='utf-8'))
                 for _, r in group.iterrows()}
        for steps in (50, 100):
            a, b = cases[20], cases[steps]
            schedules.append(dict(test_sayisi=size, veri_seed=seed, algoritma_seed=start, hedef_iterasyon=steps,
                                  baslangic_esit=bool(np.isclose(a['trace']['initial_fitness'], b['trace']['initial_fitness'])),
                                  ilk20_egri_esit=bool(np.allclose(a['curve'], b['curve'][:20], rtol=1e-10, atol=1e-12))))
    schedule_checks = pd.DataFrame(schedules)
    atomic_csv(Path(folder) / 'takvim_iz_kontrol.csv', schedule_checks)
    paragraphs = [f'# İterasyon deneyi\n\n{manifest["completed"]}/{manifest["total"]} IBMGO koşumu tamamlandı.', SCHEDULE_NOTE]
    paragraphs.append('Koddan doğrulama `kod/algoritmalar/bmgo.py` içinde yapılır. '
                      '`a = -1 - iteration / max_iter` ve `f_vector` içindeki `iteration / max_iter` '
                      'hareketleri toplam iterasyona bağlar. `progress = iteration / max(1, max_iter - 1)` '
                      'adaptif bit değişim oranını belirler. `_reference_mgo_coefficients` içindeki '
                      '`(iteration + 1) / max_iter` de referans hareket kontrolünü değiştirir.')
    paragraphs.append(f'Eşleşen ilk 20 iterasyon eğrisi {int(schedule_checks.ilk20_egri_esit.sum())}/'
                      f'{len(schedule_checks)} karşılaştırmada aynı kaldı. Bu gözlem tek başına '
                      'rastgele yolun bütünüyle aynı olduğu anlamına gelmez; kontrol takvimi toplam bütçeye bağlıdır.')
    for _, r in paired.iterrows():
        paragraphs.append(f'{int(r.test_sayisi)} testte 20→{int(r.hedef_iterasyon)} geçişinde ortalama puan farkı '
                          f'{r.fark_mean:.6f} (20 − yeni). {int(r.iyilesti)} çift iyileşti, {int(r.esit)} eşit kaldı, '
                          f'{int(r.kotulesti)} çift kötüleşti. Ortalama ek arama {r.ek_arama_mean:.2f} sn, '
                          f'süre çarpanı {r.arama_carpani_mean:.2f}.')
    paragraphs += ['Bulgular bu sentetik havuzlar, başlangıçlar ve çalışma ortamına aittir. '
                   'Ortalama farklar tek başına anlamlı üstünlük veya evrensel bir iterasyon sınırı göstermez.',
                   'Yakınsama görselleri veri tohumu bazındadır. Her çizgi aynı veride beş başlangıcın ortalaması, '
                   'bant minimum–maksimum aralığıdır. Eğriler gerçek çağrı numaralarına yerleştirilmiştir.']
    (Path(folder) / 'RAPOR.md').write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
    return summary, paired


def generate_scale(folder):
    rows, manifest = load_records(folder)
    summary = aggregate(rows, ['test_sayisi', 'yontem'])
    summary['sirala'] = summary.yontem.map({'IBMGO': 0, 'GA': 1, 'Greedy': 2})
    summary = summary.sort_values(['test_sayisi', 'sirala']).drop(columns='sirala')
    atomic_csv(Path(folder) / 'olcek_ozet.csv', summary)
    body = []
    for _, r in summary.iterrows():
        body.append([int(r.test_sayisi), r.yontem, int(r.kosum_sayisi), _pm(r.fitness_mean, r.fitness_std, 4),
                     _pm(r.secilen_test_mean, r.secilen_test_std, 1),
                     _pm(r.secili_paket_sn_mean / 60, r.secili_paket_sn_std / 60, 1),
                     _pm(r.sure_tasarrufu_yuzde_mean, r.sure_tasarrufu_yuzde_std, 2),
                     _pm(r.arama_sn_mean, r.arama_sn_std, 2)])
    table_slide(folder, '01_olcek_ozet.png', 'Büyük test havuzlarında yöntem karşılaştırması',
                'IBMGO ve GA her büyüklükte 9 koşum · Greedy her büyüklükte 3 koşum',
                ['Test', 'Yöntem', 'Koşum', 'Amaç puanı ↓\nort. ± std', 'Seçilen test\nort. ± std',
                 'Paket · dk\nort. ± std', 'Tasarruf · %\nort. ± std', 'Arama · sn\nort. ± std'], body,
                [.065, .09, .065, .18, .145, .145, .16, .15],
                '3 veri tohumu; IBMGO/GA veri başına 3 başlangıç, Greedy veri başına tek koşum.\n'
                'Standart sapma örnek standart sapmasıdır. Puanları aynı büyüklük içinde karşılaştırın.')
    fig = _slide('Büyük havuzlarda paket seçmek ne kadar sürdü?',
                 'Çubuklar ortalama, hata çizgileri örnek standart sapma · Gerçek duvar saati')
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.075, right=.97, bottom=.21, top=.77, wspace=.25)
    for ax, size in zip(axes, sorted(rows.test_sayisi.unique())):
        part = summary[summary.test_sayisi == size]
        ax.bar(part.yontem, part.arama_sn_mean, color=[METHOD_COLORS[m] for m in part.yontem],
               yerr=part.arama_sn_std, capsize=6, width=.55)
        ceiling = (part.arama_sn_mean + part.arama_sn_std).max()
        for position, (_, row) in enumerate(part.iterrows()):
            ax.text(position, row.arama_sn_mean + row.arama_sn_std + ceiling * .04,
                    f'{row.arama_sn_mean:.2f} sn', ha='center', fontsize=17, color=INK)
        ax.set(title=f'{size} test', ylabel='Arama süresi · saniye', ylim=(0, ceiling * 1.35))
    _style(axes)
    fig.text(.05, .12, 'Arama süresi, seçilen paketin tahmini çalıştırma süresinden ayrı ölçülür.', fontsize=17, color=GRAY)
    _export(fig, folder, '02_arama_suresi.png')
    structure = pd.read_csv(Path(folder) / 'havuz_yapisi.csv')
    body = [[int(r.test_sayisi), int(r.veri_seed), int(r.kapsam_hedefi), int(r.kritik_hedef), int(r.on_kosul_baglantisi),
             int(r.hedef_destegi_min), f'{r.hedef_destegi_medyan:g}', int(r.hedef_destegi_max)] for _, r in structure.iterrows()]
    table_slide(folder, '03_havuz_yapisi.png', 'Havuz büyüyünce kapsam yapısı nasıl değişti?',
                'Her hedef için onu kapsayan test sayısı da ölçüldü',
                ['Test', 'Veri\ntohumu', 'Kapsam\nhedefi', 'Kritik\nhedef', 'Ön koşul\nbağlantısı',
                 'Kapsayan test\nen az', 'Kapsayan test\nmedyan', 'Kapsayan test\nen çok'], body,
                [.09, .10, .12, .12, .15, .14, .14, .14],
                'Büyüklük tek başına zorluk göstergesi değildir. Hedefler ve aynı hedefe ait alternatifler birlikte incelenir.\n'
                'Destek istatistikleri bütün kapsam hedefleri üzerinden hesaplanır.')
    checks = summary[['test_sayisi', 'yontem', 'kosum_sayisi', 'gecerli_oran', 'kritik_kapsam_yuzde_min',
                      'genel_kapsam_yuzde_min', 'on_kosul_ihlali_max']]
    atomic_csv(Path(folder) / 'kisit_kontrolleri.csv', checks)
    body = [[int(r.test_sayisi), r.yontem, int(r.kosum_sayisi), f'%{100 * r.gecerli_oran:.0f}',
             f'%{r.kritik_kapsam_yuzde_min:.2f}', f'%{r.genel_kapsam_yuzde_min:.2f}', int(r.on_kosul_ihlali_max)]
            for _, r in checks.iterrows()]
    table_slide(folder, '04_kisit_kontrolleri.png', 'Seçilen paketler geçerli mi?',
                'Her yöntem ve büyüklükte en zayıf kapsam ile en yüksek ihlal kontrol edildi',
                ['Test', 'Yöntem', 'Koşum', 'Geçerli\npaket', 'Kritik kapsam\nen az', 'Genel kapsam\nen az',
                 'Ön koşul ihlali\nen fazla'], body, [.1, .13, .1, .15, .17, .17, .18],
                'Geçerli paket için kritik kapsam %100, genel kapsam en az %90 ve ön koşul ihlali sıfır olmalı.')
    paragraphs = [f'# Ölçek deneyi\n\n{manifest["completed"]}/{manifest["total"]} koşum tamamlandı. '
                  '18 IBMGO, 18 GA ve 6 Greedy kaydı ayrı tutuldu.']
    for size, group in summary.groupby('test_sayisi'):
        lowest = group.fitness_mean.min()
        best = [r.yontem for _, r in group.iterrows() if np.isclose(r.fitness_mean, lowest, rtol=1e-10, atol=1e-12)
                or f'{r.fitness_mean:.5f}' == f'{lowest:.5f}']
        parts = [f'{r.yontem} amaç puanı {r.fitness_mean:.5f} ± {r.fitness_std:.5f}, '
                 f'arama {r.arama_sn_mean:.2f} sn, paket {r.secili_paket_sn_mean / 60:.1f} dk'
                 for _, r in group.iterrows()]
        paragraphs.append(f'{int(size)} testte ' + '; '.join(parts) + f'. En düşük gösterilen ortalama puan {", ".join(best)} ile bulundu.')
    paragraphs.append('Tasarruflar sentetik test sürelerinden hesaplanır. Bu çalışma büyük havuzlardaki '
                      'paket kalitesini ve hesaplama maliyetini ölçer. Test sayısının artması tek başına zorluk kanıtı değildir.')
    for seed, group in structure.groupby('veri_seed'):
        a, b = group.sort_values('test_sayisi').iloc[[0, -1]].to_dict('records')
        paragraphs.append(f'Veri {seed} için 2000→5000 testte hedef sayısı {a["kapsam_hedefi"]}→{b["kapsam_hedefi"]}, '
                          f'hedef başına medyan destek {a["hedef_destegi_medyan"]:g}→{b["hedef_destegi_medyan"]:g} oldu.')
    (Path(folder) / 'RAPOR.md').write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['iteration', 'scale', 'all'], default='all')
    parser.add_argument('--iteration-dir', type=Path, default=ROOT / 'sonuclar/ek_iterasyon_2026-11-13')
    parser.add_argument('--scale-dir', type=Path, default=ROOT / 'sonuclar/ek_olcek_2026-11-13')
    args = parser.parse_args()
    if args.kind in {'iteration', 'all'}:
        generate_iteration(args.iteration_dir)
    if args.kind in {'scale', 'all'}:
        generate_scale(args.scale_dir)


if __name__ == '__main__':
    main()
