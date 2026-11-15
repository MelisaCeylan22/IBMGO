"""Uyum denetiminden geçen iki iterasyon deneyini salt okunur kaynaklardan özetle."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from kod.deneyler import ek_deneyler as base
from kod.deneyler.sunum_deneyi import Settings, digest_pool, make_pool
from kod.raporlama import ek_deney_raporu as style


TRANSITIONS = ((20, 50), (20, 100), (50, 100))
GROUPS = {'250_500': (250, 500), '1000_5000': (1000, 5000)}


def transition_pairs(rows):
    keys = ['test_sayisi', 'veri_seed', 'algoritma_seed']
    if rows.duplicated(keys + ['iterasyon']).any():
        raise ValueError('Aynı koşum birden fazla kez bulunuyor.')
    records = []
    for previous, following in TRANSITIONS:
        a = rows[rows.iterasyon == previous].set_index(keys)
        b = rows[rows.iterasyon == following].set_index(keys)
        if set(a.index) != set(b.index):
            raise ValueError('Geçiş için eşleşen veri ve başlangıçlar eksik.')
        for key, old in a.iterrows():
            new = b.loc[key]
            delta = float(old.fitness - new.fitness)
            equal = np.isclose(old.fitness, new.fitness, rtol=1e-10, atol=1e-12)
            outcome = 'eşit' if equal else 'iyileşti' if delta > 0 else 'kötüleşti'
            records.append(dict(test_sayisi=key[0], veri_seed=key[1], algoritma_seed=key[2],
                                onceki_iterasyon=previous, yeni_iterasyon=following,
                                fitness_onceki=old.fitness, fitness_yeni=new.fitness,
                                fitness_farki=delta, sonuc=outcome,
                                tasarruf_farki_yuzde_puan=new.sure_tasarrufu_yuzde - old.sure_tasarrufu_yuzde,
                                ek_arama_sn=new.arama_sn - old.arama_sn,
                                arama_carpani=new.arama_sn / old.arama_sn))
    return pd.DataFrame(records)


def summarize_transitions(pairs):
    rows = []
    for key, group in pairs.groupby(['test_sayisi', 'onceki_iterasyon', 'yeni_iterasyon']):
        rows.append(dict(test_sayisi=key[0], onceki_iterasyon=key[1], yeni_iterasyon=key[2],
                         cift=len(group), fitness_farki_mean=group.fitness_farki.mean(),
                         fitness_farki_std=group.fitness_farki.std(),
                         iyilesti=int((group.sonuc == 'iyileşti').sum()),
                         esit=int((group.sonuc == 'eşit').sum()), kotulesti=int((group.sonuc == 'kötüleşti').sum()),
                         tasarruf_pp_mean=group.tasarruf_farki_yuzde_puan.mean(),
                         tasarruf_pp_std=group.tasarruf_farki_yuzde_puan.std(),
                         ek_arama_sn_mean=group.ek_arama_sn.mean(), ek_arama_sn_std=group.ek_arama_sn.std(),
                         arama_carpani_mean=group.arama_carpani.mean()))
    return pd.DataFrame(rows)


def _validate_source(folder, sizes):
    rows, manifest = style.load_records(folder)
    plan = manifest['identity']['plan']
    required = dict(sizes=list(sizes), data_seeds=[83, 97, 109], algorithm_seeds=[42, 43, 44, 45, 46],
                    iterations=[20, 50, 100], population=12, local_trials=12, methods=['IBMGO'], smoke=False)
    if any(plan.get(k) != v for k, v in required.items()) or len(rows) != 90:
        raise ValueError('Kaynak deneyin tasarımı beklenen 90 IBMGO koşumuyla uyuşmuyor.')
    for relative, digest in manifest['version']['hashes'].items():
        if hashlib.sha256((base.ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Kayıtlı hesaplama kodu değişmiş. {relative}')
    for relative, digest in plan.get('extension_code', {}).items():
        if hashlib.sha256((base.ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Genişletme çalıştırıcısının kaynak kimliği değişmiş.')
    pools = {}
    for n in sizes:
        for seed in plan['data_seeds']:
            with np.load(folder / 'girdiler' / f'{n}_{seed}.npz') as saved:
                stored = dict(saved)
            expected = manifest['identity']['inputs'][f'{n}_{seed}']
            if digest_pool(stored) != expected or digest_pool(make_pool(n, seed)) != expected:
                raise ValueError(f'Girdi kimliği veya veri üretimi uyuşmuyor. {n}/{seed}')
            pools[n, seed] = stored
    for _, row in rows.iterrows():
        payload = json.loads((folder / 'kosumlar' / f'{row.kosum}.json').read_text(encoding='utf-8'))
        config = Settings(sizes=[int(row.test_sayisi)], data_seeds=[int(row.veri_seed)],
                          algorithm_seeds=[int(row.algoritma_seed)], iterations=int(row.iterasyon))
        settings = payload['settings']
        if (settings['sizes'] != [int(row.test_sayisi)] or settings['data_seeds'] != [int(row.veri_seed)]
                or settings['algorithm_seeds'] != [int(row.algoritma_seed)]
                or settings['iterations'] != int(row.iterasyon) or payload['trace']['steps'] != int(row.iterasyon)):
            raise ValueError('Koşumun boyut, tohum veya iterasyon kimliği uyuşmuyor.')
        for field in ('min_coverage', 'w_time', 'w_count', 'w_redundancy', 'w_rare',
                      'population', 'local_trials', 'adaptive_policy'):
            if settings[field] != getattr(config, field):
                raise ValueError(f'Kaynak ayarları uyuşmuyor. {row.kosum}/{field}')
        if payload['resolved_options'] != config.options(int(row.test_sayisi)):
            raise ValueError('Boyuta duyarlı politika uyuşmuyor.')
        if payload['budget'] != config.budget(int(row.test_sayisi)):
            raise ValueError('Değerlendirme bütçesi uyuşmuyor.')
        if payload['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Koşumun kaynak deney kimliği uyuşmuyor.')
        if row.girdi_kimligi != manifest['identity']['inputs'][f'{int(row.test_sayisi)}_{int(row.veri_seed)}']:
            raise ValueError('Koşumun girdi kimliği uyuşmuyor.')
        for field, value in payload['record'].items():
            actual = row[field]
            if isinstance(value, (int, float, bool)):
                if not np.isclose(actual, value, rtol=1e-12, atol=1e-12):
                    raise ValueError(f'CSV ve koşum kaydı farklı. {row.kosum}/{field}')
            elif actual != value:
                raise ValueError(f'CSV ve koşum kaydı farklı. {row.kosum}/{field}')
        base.verify_trace(payload['trace'], payload['curve'], row.fitness, row.planlanan_degerlendirme)
        for field, trace_field in (
                ('baslangic_fitness', 'initial_fitness'), ('gercek_degerlendirme', 'evaluations'),
                ('ilk_en_iyi_adim', 'first_final_step'), ('ilk_en_iyi_degerlendirme', 'first_final_evaluation')):
            if not np.isclose(row[field], payload['trace'][trace_field], rtol=1e-12, atol=1e-12):
                raise ValueError(f'Özet ile gerçek yakınsama izi farklı. {row.kosum}/{field}')
        pool = pools[int(row.test_sayisi), int(row.veri_seed)]
        selected = np.asarray(payload['selection'], dtype=int)
        if selected.shape != (int(row.test_sayisi),) or not np.isin(selected, [0, 1]).all():
            raise ValueError('Kayıtlı seçimin boyutu veya değerleri geçersiz.')
        metrics = base.get_suite_metrics(selected, pool['coverage'], pool['durations'],
                                         pool['dependencies'], pool['critical'])
        recalculated = dict(
            fitness=base.calculate_fitness(selected, coverage_matrix=pool['coverage'],
                durations=pool['durations'], dependency_matrix=pool['dependencies'],
                critical_feature_indices=pool['critical'], min_coverage=config.min_coverage,
                w_time=config.w_time, w_count=config.w_count, w_redundancy=config.w_redundancy,
                w_rare=config.w_rare),
            secilen_test=metrics['selected_count'], tam_paket_sn=metrics['total_duration'],
            secili_paket_sn=metrics['selected_duration'], sure_tasarrufu_yuzde=100 * (1 - metrics['time_ratio']),
            kritik_kapsam_yuzde=100 * (1 - metrics['critical_missing'] / metrics['critical_total']),
            genel_kapsam_yuzde=100 * metrics['coverage_ratio'], on_kosul_ihlali=metrics['broken_dependencies'])
        if any(not np.isclose(row[k], value, rtol=1e-10, atol=1e-12) for k, value in recalculated.items()):
            raise ValueError(f'Kayıtlı paket ölçümleri yeniden hesaplanan değerlerle uyuşmuyor. {row.kosum}')
        valid = not metrics['critical_missing'] and not metrics['broken_dependencies'] and metrics['coverage_ratio'] >= .9 - 1e-12
        if bool(row.gecerli) != bool(valid):
            raise ValueError('Geçerlilik etiketi gerçek kapsam ve ön koşul kontrolleriyle uyuşmuyor.')
    counts = rows.groupby(['test_sayisi', 'iterasyon']).size()
    if not (counts == 15).all():
        raise ValueError('Her büyüklük/iterasyon için 15 kayıt bulunmalı.')
    return rows, manifest


def combine_sources(old_folder, new_folder, output):
    old_folder, new_folder, output = map(lambda p: Path(p).resolve(), (old_folder, new_folder, output))
    if any(output == p or p in output.parents for p in (old_folder, new_folder)):
        raise ValueError('Birleşik rapor kaynak klasörlerinden ayrı olmalı.')
    old, a = _validate_source(old_folder, (250, 500))
    new, b = _validate_source(new_folder, (1000, 5000))
    for key in ('code_hashes', 'python', 'numpy', 'pandas'):
        if a['identity'][key] != b['identity'][key]:
            raise ValueError(f'Kaynaklar açıklamasız birleştirilemez. Farklı {key}.')
    for relative, digest in b['identity']['plan']['preserved_experiments'].items():
        if hashlib.sha256((base.ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Önceki deney kayıtlarından biri değişmiş.')
    rows = pd.concat([old.assign(kaynak_deney=old_folder.name), new.assign(kaynak_deney=new_folder.name)], ignore_index=True)
    if len(rows) != 180 or rows.kosum.duplicated().any():
        raise ValueError('Birleşik tasarım tam 180 farklı koşum içermeli.')
    manifest = dict(status='complete', total=180, completed=180, compatible=True,
                    created_utc=base.timestamp(), sources=[dict(folder=str(p.relative_to(base.ROOT)),
                    fingerprint=m['fingerprint'], runs=len(r)) for p, m, r in ((old_folder, a, old), (new_folder, b, new))],
                    computation_code_hashes=a['identity']['code_hashes'],
                    checks=['Hesaplama dosyalarının SHA-256 özetleri aynı',
                            'Python, NumPy ve pandas sürümleri aynı',
                            'Ağırlıklar, kapsam eşiği, popülasyon ve yerel arama aynı',
                            'Her büyüklükte aynı boyuta duyarlı politika doğrulandı',
                            'Üretilen ve kayıtlı girdilerin kimlikleri eşleşti',
                            'CSV satırları, bireysel kayıtlar ve gerçek izler doğrulandı',
                            'Kayıtlı seçimlerden amaç puanı, süre ve kısıtlar yeniden doğrulandı',
                            'Önceki ölçek deneyinin kayıtları yeni koşum olarak kullanılmadı'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if previous['sources'] != manifest['sources']:
            raise ValueError('Birleşik klasör daha önce farklı kaynaklarla oluşturulmuş.')
    base.atomic_json(output / 'manifest.json', manifest)
    base.atomic_csv(output / 'ham_sonuclar.csv', rows)
    return rows, manifest


def _payload(manifest, row):
    source = next(s for s in manifest['sources'] if Path(s['folder']).name == row.kaynak_deney)
    path = base.ROOT / source['folder'] / 'kosumlar' / f'{row.kosum}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def convergence_slide(output, manifest, rows, size, seed):
    fig = style._slide(f'{size} test · Veri tohumu {seed} · Yakınsama',
                       'Aynı veri üzerinde 5 başlangıç · Çizgi ortalama, bant minimum–maksimum')
    ax = fig.add_axes([.085, .26, .87, .50])
    for steps, group in rows[(rows.test_sayisi == size) & (rows.veri_seed == seed)].groupby('iterasyon'):
        events = [_payload(manifest, row)['trace']['events'] for _, row in group.iterrows()]
        traces = [np.array([e['best'] for e in trace]) for trace in events]
        evaluations = [np.array([e['evaluation'] for e in trace]) for trace in events]
        common = min(map(len, traces))
        values = np.stack([trace[:common] for trace in traces])
        # Gerçek çağrı kimliklerini kullan; ortak aralık dışında ortalama uzatma.
        x = evaluations[0][:common]
        if any(not np.array_equal(x, counts[:common]) for counts in evaluations[1:]):
            raise ValueError('Yakınsama izlerinin değerlendirme ölçekleri eşleşmiyor.')
        color = style.ITER_COLORS[int(steps)]
        ax.step(x, values.mean(axis=0), where='post', color=color, lw=2.4, label=f'{steps} iterasyon')
        ax.fill_between(x, values.min(axis=0), values.max(axis=0), step='post', color=color, alpha=.14)
        for trace, counts in zip(traces, evaluations):
            if len(trace) > common:
                ax.step(counts[common - 1:], trace[common - 1:], where='post', color=color, alpha=.25)
    ax.set(xlabel='Gerçek kümülatif fitness değerlendirmesi', ylabel='Amaç puanı · düşük daha iyi', xlim=(0, None))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.legend(frameon=False, fontsize=17)
    style._style([ax])
    fig.text(.05, .14, 'Minimum–maksimum bandı dağılımı gösterir. Güven aralığı değildir.', fontsize=16, color=style.GRAY)
    fig.text(.05, .085, 'Biten koşum ileri uzatılmaz. Yumuşatma yoktur. Kontrol takvimi toplam iterasyona bağlıdır.', fontsize=16, color=style.GRAY)
    style._export(fig, output, f'02_yakinsama_{size}_veri{seed}.png')


def summary_slide(output, summary, group_name, sizes):
    body = []
    for _, r in summary[summary.test_sayisi.isin(sizes)].iterrows():
        evaluations = f'{r.gercek_degerlendirme_mean:.0f}' if r.gercek_degerlendirme_std < 1e-10 else style._pm(
            r.gercek_degerlendirme_mean, r.gercek_degerlendirme_std, 1)
        body.append([int(r.test_sayisi), int(r.iterasyon), int(r.kosum_sayisi), style._pm(r.fitness_mean, r.fitness_std, 4),
                     f'{r.secilen_test_mean:.1f}', f'{r.sure_tasarrufu_yuzde_mean:.2f}',
                     f'{r.arama_sn_mean:.2f}', evaluations, f'%{100 * r.gecerli_oran:.0f}'])
    style.table_slide(output, f'01_iterasyon_ozet_{group_name}.png', f'İterasyon bütçesi · {sizes[0]} ve {sizes[1]} test',
                      'Her satır 3 veri tohumu × 5 algoritma başlangıcı = 15 koşum',
                      ['Test', 'İterasyon', 'Koşum', 'Amaç puanı ↓\nort. ± std', 'Seçilen\ntest · ort.',
                       'Tahmini\ntasarruf %\nort.', 'Arama\nsn · ort.', 'Gerçek\ndeğerl. · ort.', 'Geçerli\npaket'], body,
                      [.06, .085, .065, .19, .11, .12, .10, .14, .13],
                      'Standart sapma örnek standart sapmasıdır. Karşılaştırmalar aynı test büyüklüğü içindedir.\n'
                      'Daha yüksek iterasyon farklı toplam bütçe ve buna bağlı kontrol takvimi anlamına gelir.')


def benefit_slide(output, paired, group_name, sizes):
    body = []
    for _, r in paired[paired.test_sayisi.isin(sizes)].iterrows():
        body.append([int(r.test_sayisi), f'{int(r.onceki_iterasyon)}→{int(r.yeni_iterasyon)}', int(r.cift),
                     style._pm(r.fitness_farki_mean, r.fitness_farki_std, 5),
                     f'{int(r.iyilesti)} / {int(r.esit)} / {int(r.kotulesti)}',
                     style._pm(r.tasarruf_pp_mean, r.tasarruf_pp_std, 3),
                     style._pm(r.ek_arama_sn_mean, r.ek_arama_sn_std, 2), f'{r.arama_carpani_mean:.2f}×'])
    style.table_slide(output, f'03_ek_fayda_{group_name}.png', f'Ek fayda ve maliyet · {sizes[0]} ve {sizes[1]} test',
                      'Her geçişte 15 aynı veri / başlangıç çifti eşleştirilir',
                      ['Test', 'Geçiş', 'Çift', 'Δ amaç puanı\nort. ± std', 'İyileşti / eşit\n/ kötüleşti',
                       'Δ tasarruf\nyüzde puan', 'Ek arama · sn\nort. ± std', 'Arama\nçarpanı · ort.'], body,
                      [.065, .09, .055, .22, .13, .17, .16, .11],
                      'Δ amaç puanı = önceki fitness − yeni fitness. Pozitif değer iyileşmedir.\n'
                      'Δ tasarruf = yeni tasarruf − önceki tasarruf, yüzde puan. Amaç puanı ayrı bir ölçüttür.')


def paired_plot(output, pairs, size):
    fig = style._slide(f'{size} test · Ek bütçenin iki ayrı sonucu',
                       'Her nokta eşleşmiş bir koşum çifti · Elmaslar ortalama')
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.12, right=.97, top=.78, bottom=.24, wspace=.38)
    colors = ['#8250B5', '#C75A15', '#546F87']
    for ax, field, title, ylabel in [
            (axes[0], 'fitness_farki', 'Birleşik amaç puanındaki değişim', 'Önceki fitness − yeni fitness'),
            (axes[1], 'tasarruf_farki_yuzde_puan', 'Süre tasarrufundaki değişim', 'Yeni − önceki tasarruf · yüzde puan')]:
        for i, (previous, following) in enumerate(TRANSITIONS):
            values = pairs[(pairs.test_sayisi == size) & (pairs.onceki_iterasyon == previous)
                           & (pairs.yeni_iterasyon == following)][field]
            ax.scatter(i + np.linspace(-.1, .1, len(values)), values, s=43, alpha=.65, color=colors[i])
            ax.scatter([i], [values.mean()], s=85, marker='D', color=style.INK, edgecolor='white', zorder=4)
        ax.axhline(0, color=style.GRAY, ls='--', lw=1.2)
        ax.set(title=title, xticks=[0, 1, 2], xticklabels=['20→50', '20→100', '50→100'], ylabel=ylabel)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    style._style(axes)
    fig.text(.05, .13, 'Pozitif fark ilgili ölçütte iyileşme gösterir. Birleşik puan farkı süre tasarrufu değildir.', fontsize=16, color=style.GRAY)
    fig.text(.05, .075, 'Eşitlik toleransı rtol=1e−10, atol=1e−12. Sayısal farklar genel üstünlük kanıtı olarak yorumlanmaz.', fontsize=16, color=style.GRAY)
    style._export(fig, output, f'04_eslesmis_farklar_{size}.png')


def generate(old_folder, new_folder, output):
    output = Path(output)
    rows, manifest = combine_sources(old_folder, new_folder, output)
    summary = style.aggregate(rows, ['test_sayisi', 'iterasyon'])
    pairs = transition_pairs(rows)
    paired = summarize_transitions(pairs)
    base.atomic_csv(output / 'iterasyon_ozet.csv', summary)
    base.atomic_csv(output / 'eslesmis_farklar.csv', pairs)
    base.atomic_csv(output / 'eslesmis_ozet.csv', paired)
    checks = summary[['test_sayisi', 'iterasyon', 'kosum_sayisi', 'gecerli_oran',
                      'kritik_kapsam_yuzde_min', 'genel_kapsam_yuzde_min', 'on_kosul_ihlali_max']]
    base.atomic_csv(output / 'kisit_kontrolleri.csv', checks)
    for name, sizes in GROUPS.items():
        summary_slide(output, summary, name, sizes)
        benefit_slide(output, paired, name, sizes)
    for size in sorted(rows.test_sayisi.unique()):
        paired_plot(output, pairs, int(size))
        for seed in sorted(rows.veri_seed.unique()):
            convergence_slide(output, manifest, rows, int(size), int(seed))
    paragraphs = ['# Genişletilmiş iterasyon deneyi',
                  '90 eski ve 90 yeni IBMGO koşumu uyum kontrolünden sonra 180 koşum olarak özetlendi. '
                  'Yeni koşumlar 1000 ve 5000 testte gerçekten yeniden çalıştırıldı.',
                  style.SCHEDULE_NOTE,
                  'Hesaplama kaynaklarının içerik özetleri ve kütüphane sürümleri aynı. '
                  'Ağırlıklar, onarım, veri üretimi ve boyuta duyarlı politika korunmuştur. '
                  'Yeni çalıştırıcı yalnızca boyut planını ve kayıt akışını genişletir.',
                  'Tahmini süre tasarrufu sentetik test sürelerinden hesaplanır. Arama süresi bu bilgisayarda, '
                  'gerçek iz kaydı açıkken ölçülmüştür; gerçek bir CI/CD hızlanması ölçümü değildir.']
    for size, part in paired.groupby('test_sayisi'):
        paragraphs.append(f'## {size} test')
        means = summary[summary.test_sayisi == size]
        paragraphs.append('Ortalama amaç puanı / arama süresi: ' + '; '.join(
            f'{int(r.iterasyon)} iterasyonda {r.fitness_mean:.6f} / {r.arama_sn_mean:.2f} sn'
            for _, r in means.iterrows()) + '.')
        for _, r in part.iterrows():
            paragraphs.append(f'{int(r.onceki_iterasyon)}→{int(r.yeni_iterasyon)} geçişinde ortalama fitness farkı '
                              f'{r.fitness_farki_mean:.6f} (önceki − yeni), '
                              f'iyileşen/eşit/kötüleşen {int(r.iyilesti)}/{int(r.esit)}/{int(r.kotulesti)}. '
                              f'Tasarruf farkı {r.tasarruf_pp_mean:+.3f} yüzde puan. '
                              f'Ek arama {r.ek_arama_sn_mean:.2f} sn, eşleşmiş süre çarpanı ortalaması {r.arama_carpani_mean:.2f}.')
    paragraphs += ['Bu kayıtlarda 20 iterasyon en kısa aramayı sağlar. 50 iterasyon kalite–süre dengesi için '
                   'değerlendirilebilir. 100 iterasyonun daha düşük ortalama puanı ek süre karşılığında elde edilir; '
                   'tek tek her koşumun iyileşmesi garanti değildir.',
                   'Kısa arama ile daha yüksek bütçeli arama arasındaki seçim, bu kayıtlardaki ek kalite ve '
                   'ek hesaplama süresi birlikte değerlendirilerek yapılır. Bulgular bu havuz ve başlangıçlarla sınırlıdır.',
                   'Önceki 1212 değerlendirmeli GA kayıtları daha yüksek bütçeli IBMGO ile üstünlük kıyası için kullanılmadı. '
                   'Ölçek deneyi kendi bölümünde ve kendi kayıtlarında korunur.',
                   'CSV dosyaları ham kayıtları, 12 satırlık iterasyon özetini ve 180 eşleşmiş geçişi içerir. '
                   'Min–max bantları güven aralığı değildir; eğriler gerçek çağrılarda biter.']
    (output / 'RAPOR.md').write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
    return summary, paired


def show_group(output, prefix):
    """Kayıtlı iki tablo görünümü arasında arama başlatmadan geçiş yap."""
    import ipywidgets as w
    from IPython.display import display
    output = Path(output)
    choice = w.Dropdown(options=[('250–500 test', '250_500'), ('1000–5000 test', '1000_5000')],
                        value='1000_5000', description='Büyüklükler', style={'description_width': 'initial'})
    picture = w.Image(layout=w.Layout(width='100%', max_width='1100px'))

    def update(change=None):
        picture.value = (output / 'sunum' / f'{prefix}_{choice.value}.png').read_bytes()

    choice.observe(update, names='value')
    update()
    display(w.VBox([choice, picture]))


def show_convergence(output):
    import ipywidgets as w
    from IPython.display import display
    output = Path(output)
    size = w.Dropdown(options=[250, 500, 1000, 5000], value=1000, description='Test sayısı')
    seed = w.Dropdown(options=[83, 97, 109], value=83, description='Veri tohumu')
    picture = w.Image(layout=w.Layout(width='100%', max_width='1100px'))

    def update(change=None):
        picture.value = (output / 'sunum' / f'02_yakinsama_{size.value}_veri{seed.value}.png').read_bytes()

    size.observe(update, names='value')
    seed.observe(update, names='value')
    update()
    display(w.VBox([w.HBox([size, seed]), picture]))


def show_effects(output):
    import ipywidgets as w
    from IPython.display import display
    output = Path(output)
    paired = pd.read_csv(output / 'eslesmis_ozet.csv')
    size = w.Dropdown(options=[250, 500, 1000, 5000], value=1000, description='Test sayısı')
    picture = w.Image(layout=w.Layout(width='100%', max_width='1100px'))
    note = w.HTML()

    def update(change=None):
        picture.value = (output / 'sunum' / f'04_eslesmis_farklar_{size.value}.png').read_bytes()
        lines = []
        for _, row in paired[paired.test_sayisi == size.value].iterrows():
            lines.append(f'<p><b>{int(row.onceki_iterasyon)}→{int(row.yeni_iterasyon)}</b> · '
                         f'Δ amaç puanı {row.fitness_farki_mean:+.5f} · '
                         f'Δ tasarruf {row.tasarruf_pp_mean:+.3f} yüzde puan · '
                         f'ek arama {row.ek_arama_sn_mean:.2f} sn · '
                         f'iyileşti/eşit/kötüleşti {int(row.iyilesti)}/{int(row.esit)}/{int(row.kotulesti)}</p>')
        note.value = '<div style="font:14px/1.5 Segoe UI;color:#172c43">' + ''.join(lines) + '</div>'

    size.observe(update, names='value')
    update()
    display(w.VBox([size, picture, note]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, default=base.ROOT / 'sonuclar/ek_iterasyon_2026-11-13')
    parser.add_argument('--new', type=Path, default=base.ROOT / 'sonuclar/ek_iterasyon_1000_5000_2026-11-13')
    parser.add_argument('--output', type=Path, default=base.ROOT / 'sonuclar/iterasyon_birlesik_180_2026-11-13')
    args = parser.parse_args()
    generate(args.old, args.new, args.output)


if __name__ == '__main__':
    main()
