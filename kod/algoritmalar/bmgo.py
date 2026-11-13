"""Bağımlılık ve kapsam kısıtlı ikili Dağ Ceylanı optimizasyonu.

MGO'nun TSM/MH/BMH/MSF hareket mekanizmalarını kullanır. Sürekli aday
hareketleri V-şekilli dönüşümle mevcut 0/1 bitlerinde *değiştirme olasılığı*
olarak uygulanır. Uygunluk onarımı ve gereksiz test silme adımı eklenmiştir.
Bu nedenle özgün sürekli MGO'nun bire bir kopyası değil, probleme uyarlanmış
bir BMGO varyantıdır.
"""

from __future__ import annotations

import numpy as np

from kod.algoritmalar.fitness import _rare_target_weights, calculate_fitness


def v_shaped_transfer(x):
    """V(x) = |tanh(x)|: ikili bit değişim olasılığı."""
    return np.abs(np.tanh(x))


def s_shaped_transfer(x):
    """S(x) = 1/(1+exp(-x)): bir bitin 1 olma olasiligi."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def dimension_adaptive_settings(num_tests):
    """Keşifsel deneylerde belirlenen boyuta duyarli hibrit BMGO politikasi.

    42 ve 53 veri tohumlariyla belirlenmis on secimdir; 71 veri tohumu
    ayri dogrulamada kullanilmali. Bir evrensel MGO kuralı oldugu iddia
    edilmez. Bu kural fitness hesabina ek aday/degerlendirme katmaz.
    """
    if num_tests < 1:
        raise ValueError("num_tests pozitif olmali")
    settings = dict(
        adaptive_binary=True,
        diverse_initialization=True,
        local_search_trials=12,
    )
    if num_tests <= 250:
        return settings
    settings.update(reference_mgo_formulas=True,
                    chaotic_initialization=True, nonlinear_control=True)
    if num_tests <= 750:
        settings["local_search_strategy"] = "large"
    else:
        settings["reference_mgo_probability"] = 0.5
    return settings


def _icmic_population(rng, pop_size, dimensions):
    """ICMIC sin(0.7*pi/x) kaotik haritasiyla surekli baslangic uretir.

    Li, Geng ve Sheng (2024) makalesindeki ICMIC baslangic fikrini
    ikili test secimi icin kullaniyoruz. Sifira bolmeyi sinirliyoruz.
    """
    state = rng.uniform(-1.0, 1.0, dimensions)
    population = np.empty((pop_size, dimensions), dtype=float)
    for i in range(pop_size):
        safe = np.where(np.abs(state) < 1e-6, np.where(state < 0, -1e-6, 1e-6), state)
        state = np.sin((0.7 * np.pi) / safe)
        population[i] = state
    return population


def _reference_mgo_coefficients(rng, dimensions, iteration, max_iter, nonlinear=False):
    """MGO yazarinin Coefficient_Vector.m fonksiyonundaki dort katsayi ailesi.

    Nonlinear=True olunca literaturden esinlenen, erken hizli/gec yavas
    kontrol gecisi kullanilir; bu dal ozgun MGO'nun birebir kopyasi degildir.
    """
    progress = (iteration + 1) / max_iter
    a = -1.0 - (1.0 - (1.0 - progress) ** 3) if nonlinear else -1.0 - progress
    u = rng.standard_normal(dimensions)
    v = rng.standard_normal(dimensions)
    return np.asarray([
        rng.random(dimensions),
        np.full(dimensions, (a + 1.0) + rng.random()),
        a * rng.standard_normal(dimensions),
        u * (v ** 2) * np.cos((rng.random() * 2.0) * u),
    ])


def _repair_and_prune(
    bits, coverage, durations, critical, required_count, parents_of,
    children_of, depths,
):
    """Ön koşul ekle, kapsanmayan gereksinimi tamamla, gereksiz testleri at."""
    selected = np.asarray(bits, dtype=bool).copy()
    n, f = coverage.shape
    critical_mask = np.zeros(f, dtype=bool)
    critical_mask[critical] = True

    # Veri üreticimiz ön koşul testini daima önce konumlandırır (DAG).
    for idx in range(n - 1, -1, -1):
        if selected[idx] and len(parents_of[idx]):
            selected[parents_of[idx]] = True

    hits = np.sum(coverage[selected], axis=0).astype(int)

    def add_with_parents(idx):
        stack = [int(idx)]
        while stack:
            current = stack.pop()
            if selected[current]:
                continue
            selected[current] = True
            hits[:] += coverage[current]
            stack.extend(int(p) for p in parents_of[current])

    # Kritik gereksinimler ve ardından asgari genel kapsam doldurulur.
    for _ in range(n + 1):
        uncovered = hits == 0
        critical_missing = uncovered & critical_mask
        if not critical_missing.any() and np.count_nonzero(~uncovered) >= required_count:
            break

        weights = uncovered.astype(float)
        if critical_missing.any():
            weights[critical_mask] = 10.0 * critical_missing[critical_mask]
        gains = coverage @ weights
        gains[selected] = 0.0
        scores = gains / durations
        idx = int(np.argmax(scores))
        if scores[idx] <= 0:
            raise ValueError("Veri kumesinde gerekli kapsami saglayacak test yok")
        add_with_parents(idx)
    else:
        raise ValueError("Kapsam onarimi tamamlanamadi")

    # Önce bağımlı testleri, aynı derinlikte pahalı testleri çıkar.
    # Kısıtları her silmeden sonra tekrar denetleriz.
    order = sorted(np.flatnonzero(selected), key=lambda i: (depths[i], durations[i]), reverse=True)
    covered_count = int(np.count_nonzero(hits))
    for idx in order:
        if not selected[idx] or any(selected[child] for child in children_of[idx]):
            continue
        lost = (hits == 1) & (coverage[idx] != 0)
        if np.any(lost & critical_mask):
            continue
        if covered_count - int(np.count_nonzero(lost)) < required_count:
            continue
        selected[idx] = False
        hits -= coverage[idx]
        covered_count -= int(np.count_nonzero(lost))

    return selected.astype(np.int8)


def binary_mountain_gazelle_optimizer(
    coverage_matrix,
    durations,
    dependency_matrix=None,
    pop_size=20,
    max_iter=50,
    w_time=0.4,
    w_redundancy=0.2,
    random_seed=42,
    critical_feature_indices=None,
    min_coverage=1.0,
    w_count=0.4,
    adaptive_binary=False,
    diverse_initialization=False,
    local_search_trials=0,
    reference_mgo_formulas=False,
    chaotic_initialization=False,
    nonlinear_control=False,
    spiral_search_trials=0,
    binary_transfer="v_flip",
    reference_mgo_probability=1.0,
    flip_rate_start=0.04,
    flip_rate_end=0.02,
    local_search_strategy="swap",
    w_rare=0.0,
):
    """En iyi ikili seçimi, fitness'ı ve iterasyon eğrisini döndürür.

    Varsayılan değerler özgün BMGO deneyiyle uyumludur. İyileştirilmiş
    varyant ``adaptive_binary=True``, ``diverse_initialization=True`` ve
    ``local_search_trials>0`` kullanır. Her yerel arama denemesi ayrıca bir
    fitness değerlendirmesidir; karşılaştırmada bu bütçe sayılmalıdır.
    """
    coverage = (np.asarray(coverage_matrix) != 0).astype(np.int32)
    duration = np.asarray(durations, dtype=float)
    if coverage.ndim != 2:
        raise ValueError("coverage_matrix iki boyutlu olmali")
    n, f = coverage.shape
    if n < 1 or f < 1 or duration.shape != (n,) or np.any(duration <= 0):
        raise ValueError("Test / gereksinim boyutlari veya sureler gecersiz")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage (0,1] araliginda olmali")
    if pop_size < 2 or max_iter < 1:
        raise ValueError("pop_size >= 2 ve max_iter >= 1 olmali")
    if int(local_search_trials) != local_search_trials or local_search_trials < 0:
        raise ValueError("local_search_trials negatif olmayan tamsayi olmali")
    if int(spiral_search_trials) != spiral_search_trials or spiral_search_trials < 0:
        raise ValueError("spiral_search_trials negatif olmayan tamsayi olmali")
    if binary_transfer not in {"v_flip", "s_assign"}:
        raise ValueError("binary_transfer 'v_flip' veya 's_assign' olmali")
    if not 0.0 <= reference_mgo_probability <= 1.0:
        raise ValueError("reference_mgo_probability [0,1] araliginda olmali")
    if not (0.0 < flip_rate_end <= flip_rate_start <= 1.0):
        raise ValueError("flip_rate degerleri 0 < bitis <= baslangic <= 1 olmali")
    if local_search_strategy not in {"swap", "large"}:
        raise ValueError("local_search_strategy 'swap' veya 'large' olmali")

    critical = (
        np.arange(f, dtype=int)
        if critical_feature_indices is None
        else np.asarray(critical_feature_indices, dtype=int)
    )
    if np.any(critical < 0) or np.any(critical >= f):
        raise ValueError("Kritik gereksinim indeksi gecersiz")
    required_count = int(np.ceil(min_coverage * f - 1e-12))
    if np.count_nonzero(np.any(coverage, axis=0)) < required_count:
        raise ValueError("Veri kumesinde minimum kapsam olanaksiz")
    if not np.all(np.any(coverage[:, critical], axis=0)):
        raise ValueError("En az bir kritik gereksinimi kapsayan test yok")

    if dependency_matrix is None:
        dep = np.zeros((n, n), dtype=bool)
    else:
        dep = np.asarray(dependency_matrix) != 0
        if dep.shape != (n, n):
            raise ValueError("dependency_matrix NxN olmali")
    # Şu anki senaryolarda ebeveyn indeksleri her zaman çocuktan küçüktür.
    if np.any(np.tril(dep)):
        raise ValueError("Bagimlilik matrisi sirali ve dongusuz olmali (parent < child)")

    parent_ids, child_ids = np.nonzero(dep)
    parents_of = [np.flatnonzero(dep[:, i]) for i in range(n)]
    children_of = [np.flatnonzero(dep[i]) for i in range(n)]
    depths = np.zeros(n, dtype=int)
    for i in range(n):
        if len(parents_of[i]):
            depths[i] = 1 + int(np.max(depths[parents_of[i]]))

    rare_weights = _rare_target_weights(coverage, critical) if w_rare else None
    rng = np.random.default_rng(random_seed)
    chaotic_states = _icmic_population(rng, pop_size, n) if chaotic_initialization else None
    continuous_pop = (
        2.0 * chaotic_states if chaotic_initialization
        else rng.uniform(-2.0, 2.0, (pop_size, n))
    )
    binary_pop = np.empty((pop_size, n), dtype=np.int8)
    scores = np.empty(pop_size, dtype=float)
    inclusion_probabilities = (
        rng.permutation(np.linspace(0.04, 0.65, pop_size))
        if diverse_initialization else np.full(pop_size, 0.5)
    )
    for i in range(pop_size):
        initial_probabilities = (
            (chaotic_states[i] + 1.0) / 2.0
            if chaotic_initialization else rng.random(n)
        )
        initial = (initial_probabilities < inclusion_probabilities[i]).astype(np.int8)
        binary_pop[i] = _repair_and_prune(
            initial, coverage, duration, critical, required_count,
            parents_of, children_of, depths,
        )
        scores[i] = calculate_fitness(
            binary_pop[i], coverage, duration,
            w_time=w_time, w_redundancy=w_redundancy, w_count=w_count,
            w_rare=w_rare, _rare_weights=rare_weights,
            min_coverage=min_coverage, critical_feature_indices=critical,
            dependency_edges=(parent_ids, child_ids),
        )

    best_idx = int(np.argmin(scores))
    best_solution = binary_pop[best_idx].copy()
    best_continuous = continuous_pop[best_idx].copy()
    best_fitness = float(scores[best_idx])
    curve = np.empty(max_iter, dtype=float)

    for iteration in range(max_iter):
        a = -1.0 - iteration / max_iter
        progress = iteration / max(1, max_iter - 1)
        new_cont = []
        new_bin = []
        new_scores = []
        mean_pop = np.mean(continuous_pop, axis=0)

        for i in range(pop_size):
            current = continuous_pop[i]
            random_index = int(rng.integers(pop_size))
            random_position = continuous_pop[random_index]
            use_reference = reference_mgo_formulas and (
                reference_mgo_probability == 1.0
                or rng.random() < reference_mgo_probability
            )
            if use_reference:
                # Ozgun MGO.m: rastgele secilmis yaklasik N/3 bireyin
                # ortalamasi ve opsiyonel bir baska gazel konumu.
                p = max(1, int(np.ceil(pop_size / 3)))
                indices = rng.choice(pop_size, size=p, replace=False)
                bh = np.mean(continuous_pop[indices], axis=0)
                if rng.integers(0, 2):
                    bh = bh + continuous_pop[int(rng.integers(p - 1, pop_size))]
                cofi = _reference_mgo_coefficients(
                    rng, n, iteration, max_iter, nonlinear=nonlinear_control
                )
            else:
                bh = random_position + mean_pop * rng.random()
            f_vector = rng.standard_normal(n) * np.exp(2.0 - 2.0 * iteration / max_iter)
            if use_reference:
                c1, c2, c3, c4 = cofi[rng.integers(0, 4, size=4)]
                r1, r2, r3, r4, r5, r6 = rng.integers(1, 3, size=6)
            else:
                c1, c2, c3 = [
                    (a + 1.0) + rng.random() * a * rng.standard_normal(n)
                    for _ in range(3)
                ]
                c4 = c1
                r1, r2, r3, r4 = rng.integers(1, 3, size=4)
                r5, r6 = r1, r2

            tsm = best_continuous - np.abs((r1 * bh - r2 * current) * f_vector) * c1
            mh = (bh + c2) + (r3 * best_continuous - r4 * random_position) * c3
            d = (np.abs(current) + np.abs(best_continuous)) * (2.0 * rng.random() - 1.0)
            bmh = (current - d) + (r5 * best_continuous - r6 * bh) * c4
            msf = rng.uniform(-2.0, 2.0, n)

            for candidate in (tsm, mh, bmh, msf):
                candidate = np.clip(candidate, -4.0, 4.0)
                # V-transfer değeri "yeni bitin 1 olma olasılığı" değildir.
                # Sık kullanılan V-şekilli binary uyarlamada bit çevrilir.
                if binary_transfer == "s_assign":
                    # S-tipinde mevcut bit cevrilmez: aday surekli konumu
                    # yeni bitin dogrudan 1 olma ihtimalini belirler.
                    new_bits = rng.random(n) < s_shaped_transfer(candidate)
                else:
                    transfer = v_shaped_transfer(candidate - current)
                    if adaptive_binary:
                        # Her adayda toplam beklenen bit dönüşünü sınırla.
                        # Büyük boyutlarda çok sayıda bit çevirmek, iyi bulunan
                        # çözümün çevresinde hassas aramayı engelleyebiliyor.
                        target_flips = max(
                            2.0, n * (flip_rate_start -
                                      (flip_rate_start - flip_rate_end) * progress)
                        )
                        probabilities = np.minimum(
                            1.0, transfer * (target_flips / max(1e-12, float(transfer.sum())))
                        )
                    else:
                        probabilities = 0.4 * transfer
                    flips = rng.random(n) < probabilities
                    new_bits = np.logical_xor(binary_pop[i].astype(bool), flips)
                repaired = _repair_and_prune(
                    new_bits, coverage, duration, critical, required_count,
                    parents_of, children_of, depths,
                )
                score = calculate_fitness(
                    repaired, coverage, duration,
                    w_time=w_time, w_redundancy=w_redundancy, w_count=w_count,
                    w_rare=w_rare, _rare_weights=rare_weights,
                    min_coverage=min_coverage, critical_feature_indices=critical,
                    dependency_edges=(parent_ids, child_ids),
                )
                new_cont.append(candidate)
                new_bin.append(repaired)
                new_scores.append(score)

        all_cont = np.vstack((continuous_pop, np.asarray(new_cont)))
        all_bin = np.vstack((binary_pop, np.asarray(new_bin)))
        all_scores = np.concatenate((scores, np.asarray(new_scores)))
        survivors = np.argsort(all_scores, kind="stable")[:pop_size]
        continuous_pop = all_cont[survivors].copy()
        binary_pop = all_bin[survivors].copy()
        scores = all_scores[survivors].copy()

        if scores[0] < best_fitness:
            best_fitness = float(scores[0])
            best_solution = binary_pop[0].copy()
            best_continuous = continuous_pop[0].copy()

        # Literatürdeki komşuluk araması fikrini kısıtlı ikili probleme
        # uyarlayan deneysel seçenek: seçilmiş pahalı testten farklı bir teste
        # geç ve aynı kapsama/bağımlılık onarımından sonra sonucu değerlendir.
        # BMGO'nun TSM/MH/BMH/MSF adaylarına ek olarak sayılır.
        for _ in range(int(local_search_trials)):
            selected_ids = np.flatnonzero(best_solution)
            excluded_ids = np.flatnonzero(~best_solution.astype(bool))
            if not len(selected_ids) or not len(excluded_ids):
                break
            local = best_solution.copy()
            if local_search_strategy == "large":
                # Birden fazla pahali testin birlikte cikarilmasi, tekli
                # swap'in asamadigi kapsam kilitlerini gevsetebilir.
                leaves = np.asarray([
                    idx for idx in selected_ids
                    if not any(best_solution[child] for child in children_of[idx])
                ], dtype=int)
                pool = leaves if len(leaves) else selected_ids
                drop_count = min(len(pool), int(rng.integers(1, 4)))
                p_drop = duration[pool] / duration[pool].sum()
                drops = rng.choice(pool, size=drop_count, replace=False,
                                   p=p_drop)
                local[drops] = 0
                lost_needs = np.any(coverage[drops] != 0, axis=0)
                for _ in range(min(drop_count, 2)):
                    options = np.flatnonzero(~local.astype(bool))
                    # Yeni testler kaybedilen ozelliklerin bir kismini
                    # daha dusuk sureyle karsilamayi hedefler.
                    overlap = coverage[options] @ lost_needs.astype(np.int32)
                    weights = (overlap + 0.5) / (duration[options] + 1.0)
                    weights /= weights.sum()
                    add = int(rng.choice(options, p=weights))
                    local[add] = 1
                    lost_needs &= coverage[add] == 0
            else:
                # Sure maliyeti yuksek bir testi onun kapsamini karsilayan
                # alternatiflerden biriyle degistirme denemesi.
                selection_weights = duration[selected_ids] / duration[selected_ids].sum()
                drop = int(rng.choice(selected_ids, p=selection_weights))
                local[drop] = 0
                overlap = coverage[excluded_ids] @ coverage[drop]
                chances = (overlap + 1.0) / (duration[excluded_ids] + 1.0)
                chances = chances / np.sum(chances)
                add = int(rng.choice(excluded_ids, p=chances))
                local[add] = 1
            repaired = _repair_and_prune(
                local, coverage, duration, critical, required_count,
                parents_of, children_of, depths,
            )
            local_score = calculate_fitness(
                repaired, coverage, duration,
                w_time=w_time, w_redundancy=w_redundancy, w_count=w_count,
                w_rare=w_rare, _rare_weights=rare_weights,
                min_coverage=min_coverage, critical_feature_indices=critical,
                dependency_edges=(parent_ids, child_ids),
            )
            if local_score < best_fitness:
                best_solution = repaired.copy()
                best_fitness = float(local_score)
                best_continuous = continuous_pop[0].copy()
            worst = int(np.argmax(scores))
            if local_score < scores[worst]:
                binary_pop[worst] = repaired
                scores[worst] = local_score
                continuous_pop[worst] = np.clip(
                    best_continuous + rng.normal(0, 0.1, n), -4.0, 4.0
                )

        # Spiral bozulma: elite yaklasan surekli arama adimini ikili
        # V-transfer ile cevirir. Her deneme fitness butcesine dahildir.
        for _ in range(int(spiral_search_trials)):
            idx = int(rng.integers(pop_size))
            current = continuous_pop[idx]
            l = rng.uniform(-1.0, 1.0)
            spiral = np.clip(
                best_continuous + np.abs(best_continuous - current)
                * np.exp(l) * np.cos(2.0 * np.pi * l),
                -4.0, 4.0,
            )
            if binary_transfer == "s_assign":
                local = rng.random(n) < s_shaped_transfer(spiral)
            else:
                transfer = v_shaped_transfer(spiral - current)
                if adaptive_binary:
                    target_flips = max(
                        1.0, n * (0.625 * flip_rate_start -
                                  (0.625 * flip_rate_start - 0.5 * flip_rate_end)
                                  * progress)
                    )
                    probabilities = np.minimum(
                        1.0, transfer * (target_flips / max(1e-12, float(transfer.sum())))
                    )
                else:
                    probabilities = 0.4 * transfer
                flips = rng.random(n) < probabilities
                local = np.logical_xor(binary_pop[idx].astype(bool), flips)
            repaired = _repair_and_prune(
                local, coverage, duration, critical, required_count,
                parents_of, children_of, depths,
            )
            spiral_score = calculate_fitness(
                repaired, coverage, duration,
                w_time=w_time, w_redundancy=w_redundancy, w_count=w_count,
                w_rare=w_rare, _rare_weights=rare_weights,
                min_coverage=min_coverage, critical_feature_indices=critical,
                dependency_edges=(parent_ids, child_ids),
            )
            if spiral_score < best_fitness:
                best_fitness = float(spiral_score)
                best_solution = repaired.copy()
                best_continuous = spiral.copy()
            worst = int(np.argmax(scores))
            if spiral_score < scores[worst]:
                scores[worst] = spiral_score
                binary_pop[worst] = repaired
                continuous_pop[worst] = spiral
        curve[iteration] = best_fitness

    return best_solution, best_fitness, curve
