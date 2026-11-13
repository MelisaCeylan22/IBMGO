"""BMGO telekom regresyon deneyi: 50/100/250/500/1000 test.

Kullanim:
    pip install numpy matplotlib
    python -m kod.deneyler.main

Sonuclar varsayilan olarak ./sonuclar/nihai_bmgo klasorune kaydedilir.
Eski deney dosyalari korunur. Senaryolar sentetiktir;
gercek test kosumu veya gercek hata tespiti yapilmaz.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")  # Sunucu / VS Code ortaminda engelleyici pencere acma.
import matplotlib.pyplot as plt
import numpy as np

from kod.algoritmalar.bmgo import binary_mountain_gazelle_optimizer, dimension_adaptive_settings
from kod.algoritmalar.baselines import genetic_algorithm_optimizer, greedy_regression_selector
from kod.veri.data_generator import generate_telecom_data
from kod.algoritmalar.fitness import get_suite_metrics, prioritize_tests_and_calculate_apfd


SIZES = (50, 100, 250, 500, 1000)
OPTIMIZER_SEEDS = (42, 43, 44)
DATA_SEED = 42  # Adil karsilastirma icin her boyutta sabit veri kaynagi.
POP_SIZE = 12
MAX_ITER = 25
IMPROVED_ITER = 20
MIN_COVERAGE = 0.90
OUTPUT_DIR = "sonuclar/nihai_bmgo"
GA_GENERATIONS = MAX_ITER * 4
# BMGO: 12 + 25*4*12 = 1212 değerlendirme.
# IBMGO: 12 + 20*(4*12 + 12 yerel arama) = 1212 değerlendirme.
# GA: 12 + 100*12 = 1212 değerlendirme.
EVALUATION_BUDGET = POP_SIZE * (1 + 4 * MAX_ITER)


def early_critical_coverage(order, coverage, critical, fraction=0.25):
    """Ilk %25 testte gorulen kritik kurallar. APFD *degildir*."""
    if len(order) == 0 or len(critical) == 0:
        return 0.0
    first_count = max(1, int(np.ceil(len(order) * fraction)))
    hits = np.any(coverage[order[:first_count]][:, critical] != 0, axis=0)
    return float(100.0 * np.mean(hits))


def write_case_table(path, cases):
    """Senaryolari Excel'de okunabilecek UTF-8 CSV olarak kaydeder."""
    if not cases:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=cases[0].keys(), delimiter=";")
        writer.writeheader()
        for case in cases:
            writer.writerow({
                key: " | ".join(map(str, value)) if isinstance(value, (list, tuple)) else value
                for key, value in case.items()
            })


def make_comparison_row(algorithm, n, seed, solution, fitness_value, elapsed,
                        coverage, durations, dependencies, critical, evaluations):
    """Bütün yöntemleri aynı metriklerle değerlendirir ve denetler."""
    m = get_suite_metrics(solution, coverage, durations, dependencies, critical)
    if (m["critical_missing"] != 0 or m["broken_dependencies"] != 0
            or m["coverage_ratio"] + 1e-9 < MIN_COVERAGE):
        raise RuntimeError(f"{algorithm} gecersiz cozum: {n} test, seed={seed}")
    order, _ = prioritize_tests_and_calculate_apfd(
        solution, coverage, durations,
        dependency_matrix=dependencies,
        critical_feature_indices=critical,
    )
    return {
        "algoritma": algorithm,
        "test_sayisi": n,
        "seed": seed,
        "gereksinim_sayisi": m["total_features"],
        "secili_test": m["selected_count"],
        "baslangic_suresi_sn": round(m["total_duration"], 2),
        "secili_sure_sn": round(m["selected_duration"], 2),
        "sure_tasarrufu_yuzde": round(100 * (1 - m["time_ratio"]), 2),
        "genel_kapsam_yuzde": round(100 * m["coverage_ratio"], 2),
        "kritik_kapsam_yuzde": round(100 * (1 - m["critical_missing"] / m["critical_total"]), 2),
        "ilk_ceyrek_kritik_kapsam_yuzde": round(
            early_critical_coverage(order, coverage, critical), 2
        ),
        "kopuk_on_kosul": m["broken_dependencies"],
        "fitness": round(float(fitness_value), 6),
        "algoritma_suresi_sn": round(elapsed, 4),
        "fitness_degerlendirmesi": evaluations,
    }


def run_experiment(num_runs=3, scenario_profile="expanded", data_seed=DATA_SEED,
                   output_dir=OUTPUT_DIR):
    if num_runs < 1:
        raise ValueError("num_runs en az 1 olmali")
    optimizer_seeds = tuple(range(42, 42 + num_runs))
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    comparison_rows = []
    representative_curves = {}
    curves_1000 = {}

    print("\nBMGO - TELEKOM REGRESYON TEST SECIMI VE ONCELIKLENDIRME")
    print("Kural: kritik %100, genel en az %90; on kosullar korunur.")
    print("NOT: Veriler kurgusaldir; hata tespit matrisi yoksa APFD hesaplanmaz.\n")
    print(f"Veri profili: {scenario_profile}, veri tohumu: {data_seed}")
    print(f"BMGO, IBMGO ve GA es aday butcesi: {EVALUATION_BUDGET} fitness hesaplamasi.\n")

    for n in SIZES:
        coverage, durations, names, features, deps, metadata = generate_telecom_data(
            num_tests=n, seed=data_seed, include_metadata=True,
            scenario_profile=scenario_profile,
        )
        critical = metadata["critical_feature_indices"]
        if n == 50:
            write_case_table(
                os.path.join(output_dir, "telekom_50_ornek_senaryo.csv"),
                metadata["test_cases"],
            )

        for run_seed in optimizer_seeds:
            start = time.perf_counter()
            solution, fitness_value, convergence = binary_mountain_gazelle_optimizer(
                coverage_matrix=coverage,
                durations=durations,
                dependency_matrix=deps,
                critical_feature_indices=critical,
                min_coverage=MIN_COVERAGE,
                pop_size=POP_SIZE,
                max_iter=MAX_ITER,
                w_time=0.4,
                w_count=0.4,
                w_redundancy=0.2,
                random_seed=run_seed,
            )
            elapsed = time.perf_counter() - start
            metrics = get_suite_metrics(solution, coverage, durations, deps, critical)

            if (metrics["critical_missing"] != 0
                    or metrics["coverage_ratio"] < MIN_COVERAGE - 1e-9
                    or metrics["broken_dependencies"] != 0):
                raise RuntimeError(f"Gecersiz BMGO cozumu: {n} test, seed={run_seed}")

            order, apfd = prioritize_tests_and_calculate_apfd(
                solution, coverage, durations,
                dependency_matrix=deps,
                critical_feature_indices=critical,
            )
            early_coverage = early_critical_coverage(order, coverage, critical)

            result = {
                "test_sayisi": n,
                "seed": run_seed,
                "gereksinim_sayisi": len(features),
                "kritik_gereksinim_sayisi": len(critical),
                "yeni_is_kurali_sayisi": metadata["additional_rule_count"],
                "baglamsal_kapsam_sayisi": metadata["context_rule_count"],
                "secili_test": metrics["selected_count"],
                "elenen_test": n - metrics["selected_count"],
                "toplam_sure_sn": round(metrics["total_duration"], 2),
                "optimize_sure_sn": round(metrics["selected_duration"], 2),
                "sure_tasarrufu_yuzde": round(100 * (1 - metrics["time_ratio"]), 2),
                "genel_kapsam_yuzde": round(100 * metrics["coverage_ratio"], 2),
                "kritik_kapsam_yuzde": round(
                    100 * (1 - metrics["critical_missing"] / metrics["critical_total"]), 2
                ),
                "kopuk_on_kosul": metrics["broken_dependencies"],
                "gereksiz_kapsama_tekrari": metrics["redundant_hits"],
                "ilk_ceyrek_kritik_kural_kapsami_yuzde": round(early_coverage, 2),
                "APFD": "VERI_YOK" if apfd is None else round(apfd, 2),
                "fitness": round(fitness_value, 6),
                "algoritma_suresi_sn": round(elapsed, 4),
            }
            # Eski BMGO karsilastirmada durur; ana sonuc tablosu ve grafikler
            # asagida olusturulan uyarlanmis IBMGO cozumunu gosterecek.
            comparison_rows.append(make_comparison_row(
                "BMGO", n, run_seed, solution, fitness_value, elapsed,
                coverage, durations, deps, critical,
                EVALUATION_BUDGET,
            ))

            print(
                f"{n:4} test | {len(features):3} kapsam hedefi | seed={run_seed} "
                f"| secilen={metrics['selected_count']:3} "
                f"| kapsam=%{100 * metrics['coverage_ratio']:.1f} "
                f"| kritik=%100 | tasarruf=%{100 * (1 - metrics['time_ratio']):.1f} "
                f"| ihlal={metrics['broken_dependencies']} | {elapsed:.2f} sn"
            )

            if run_seed == optimizer_seeds[0]:
                if n == 1000:
                    curves_1000["BMGO"] = convergence
                write_case_table(
                    os.path.join(output_dir, f"telekom_{n}_bmgo_ilk_secili_testler.csv"),
                    [
                        {"oncelik_sirasi": rank, **metadata["test_cases"][int(idx)]}
                        for rank, idx in enumerate(order, 1)
                    ],
                )
                print("  Ilk BMGO'nun ilk 5 oncelikli testi (kontrol):")
                for test_idx in order[:5]:
                    print(f"    T_{test_idx + 1:04d} - {names[test_idx]}")

        # Greedy deterministiktir; bir kez calistirmak yeterli.
        start = time.perf_counter()
        greedy_sol, greedy_fit = greedy_regression_selector(
            coverage, durations, deps, critical_feature_indices=critical,
            min_coverage=MIN_COVERAGE,
        )
        greedy_elapsed = time.perf_counter() - start
        comparison_rows.append(make_comparison_row(
            "Greedy", n, "tek", greedy_sol, greedy_fit, greedy_elapsed,
            coverage, durations, deps, critical, "tek-gecis",
        ))

        # Boyuta uyarlanan IBMGO: kucuk pakette eski adaptif hareketler,
        # orta pakette referans+kaos+genis komsuluk, buyukte karma hareketler.
        # Ayarlar gelistirme tohumlarinda kesifsel olarak belirlendi;
        # bagimsiz veri tohumu ile dogrulama yapilmalidir.
        # Her dalda fitness degerlendirme butcesi aynidir.
        for run_seed in optimizer_seeds:
            start = time.perf_counter()
            improved_sol, improved_fit, improved_curve = binary_mountain_gazelle_optimizer(
                coverage_matrix=coverage,
                durations=durations,
                dependency_matrix=deps,
                pop_size=POP_SIZE,
                max_iter=IMPROVED_ITER,
                w_time=0.4,
                w_count=0.4,
                w_redundancy=0.2,
                random_seed=run_seed,
                critical_feature_indices=critical,
                min_coverage=MIN_COVERAGE,
                **dimension_adaptive_settings(n),
            )
            improved_elapsed = time.perf_counter() - start
            improved_row = make_comparison_row(
                "IBMGO", n, run_seed, improved_sol, improved_fit,
                improved_elapsed, coverage, durations, deps, critical,
                EVALUATION_BUDGET,
            )
            comparison_rows.append(improved_row)
            improved_metrics = get_suite_metrics(
                improved_sol, coverage, durations, deps, critical
            )
            improved_order, improved_apfd = prioritize_tests_and_calculate_apfd(
                improved_sol, coverage, durations,
                dependency_matrix=deps,
                critical_feature_indices=critical,
            )
            rows.append({
                "test_sayisi": n,
                "seed": run_seed,
                "gereksinim_sayisi": len(features),
                "kritik_gereksinim_sayisi": len(critical),
                "yeni_is_kurali_sayisi": metadata["additional_rule_count"],
                "baglamsal_kapsam_sayisi": metadata["context_rule_count"],
                "secili_test": improved_metrics["selected_count"],
                "elenen_test": n - improved_metrics["selected_count"],
                "toplam_sure_sn": round(improved_metrics["total_duration"], 2),
                "optimize_sure_sn": round(improved_metrics["selected_duration"], 2),
                "sure_tasarrufu_yuzde": improved_row["sure_tasarrufu_yuzde"],
                "genel_kapsam_yuzde": improved_row["genel_kapsam_yuzde"],
                "kritik_kapsam_yuzde": improved_row["kritik_kapsam_yuzde"],
                "kopuk_on_kosul": improved_row["kopuk_on_kosul"],
                "gereksiz_kapsama_tekrari": improved_metrics["redundant_hits"],
                "ilk_ceyrek_kritik_kural_kapsami_yuzde": improved_row[
                    "ilk_ceyrek_kritik_kapsam_yuzde"
                ],
                "APFD": "VERI_YOK" if improved_apfd is None else round(improved_apfd, 2),
                "fitness": improved_row["fitness"],
                "algoritma_suresi_sn": improved_row["algoritma_suresi_sn"],
            })
            if n == 1000 and run_seed == optimizer_seeds[0]:
                curves_1000["IBMGO"] = improved_curve
            if run_seed == optimizer_seeds[0]:
                representative_curves[n] = improved_curve
                write_case_table(
                    os.path.join(output_dir, f"telekom_{n}_secili_testler.csv"),
                    [
                        {"oncelik_sirasi": rank, **metadata["test_cases"][int(idx)]}
                        for rank, idx in enumerate(improved_order, 1)
                    ],
                )
                print("  IBMGO'nin ilk 5 oncelikli testi (ana sonuc):")
                for test_idx in improved_order[:5]:
                    print(f"    T_{test_idx + 1:04d} - {names[test_idx]}")
            print(
                f"  IBMGO | {n:4} test | seed={run_seed} "
                f"| secilen={improved_row['secili_test']:3} "
                f"| tasarruf=%{improved_row['sure_tasarrufu_yuzde']:.1f} "
                f"| fitness={improved_fit:.4f} | {improved_elapsed:.2f} sn"
            )

        # GA, BMGO ve IBMGO ayni fitness degerlendirme hakkina sahiptir.
        for run_seed in optimizer_seeds:
            start = time.perf_counter()
            ga_sol, ga_fit, ga_curve = genetic_algorithm_optimizer(
                coverage_matrix=coverage,
                durations=durations,
                dependency_matrix=deps,
                pop_size=POP_SIZE,
                max_generations=GA_GENERATIONS,
                random_seed=run_seed,
                critical_feature_indices=critical,
                min_coverage=MIN_COVERAGE,
            )
            ga_elapsed = time.perf_counter() - start
            ga_result = make_comparison_row(
                "GA", n, run_seed, ga_sol, ga_fit, ga_elapsed,
                coverage, durations, deps, critical,
                EVALUATION_BUDGET,
            )
            comparison_rows.append(ga_result)
            if n == 1000 and run_seed == optimizer_seeds[0]:
                curves_1000["GA"] = ga_curve
            print(
                f"  GA  | {n:4} test | seed={run_seed} "
                f"| secilen={ga_result['secili_test']:3} "
                f"| tasarruf=%{ga_result['sure_tasarrufu_yuzde']:.1f} "
                f"| fitness={ga_fit:.4f} | {ga_elapsed:.2f} sn"
            )
        print(f"  Greedy | {n:4} test | fitness={greedy_fit:.4f} | {greedy_elapsed:.2f} sn\n")

    csv_path = os.path.join(output_dir, "telekom_bmgo_denemeler.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    for n, curve in representative_curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=f"{n} test")
    ax.set_title("Uyarlanmis IBMGO Yakinsama Grafigi - Telekom Testleri")
    ax.set_xlabel("Iterasyon")
    ax.set_ylabel("Fitness (dusuk daha iyi)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bmgo_yakinsama.png"), dpi=200)
    plt.close(fig)

    grouped = {
        n: [r["sure_tasarrufu_yuzde"] for r in rows if r["test_sayisi"] == n]
        for n in SIZES
    }
    means = [np.mean(grouped[n]) for n in SIZES]
    stds = [np.std(grouped[n], ddof=1) if len(grouped[n]) > 1 else 0 for n in SIZES]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(SIZES, means, yerr=stds, marker="o", capsize=4)
    ax.set_title(f"Test Paketi Boyutuna Gore Sure Tasarrufu ({num_runs} Kosum)")
    ax.set_xlabel("Baslangictaki test sayisi")
    ax.set_ylabel("Ortalama sure tasarrufu (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bmgo_sure_tasarrufu.png"), dpi=200)
    plt.close(fig)

    print(f"\nCSV: {csv_path}")
    print(f"Senaryo tablolari: {output_dir}/telekom_50_ornek_senaryo.csv ve secili testler")
    print(f"Grafikler: {output_dir}/bmgo_yakinsama.png, {output_dir}/bmgo_sure_tasarrufu.png")

    comparison_path = os.path.join(output_dir, "algoritma_karsilastirma.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(comparison_rows)

    algorithm_names = ("Greedy", "GA", "BMGO", "IBMGO")
    summary_rows = []
    for n in SIZES:
        for name in algorithm_names:
            group = [
                row for row in comparison_rows
                if row["test_sayisi"] == n and row["algoritma"] == name
            ]

            def avg(key):
                return float(np.mean([float(row[key]) for row in group]))

            def std(key):
                values = [float(row[key]) for row in group]
                return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            summary_rows.append({
                "test_sayisi": n,
                "gereksinim_sayisi": group[0]["gereksinim_sayisi"],
                "algoritma": name,
                "kosum_sayisi": len(group),
                "ortalama_secili_test": round(avg("secili_test"), 2),
                "ortalama_tasarruf_yuzde": round(avg("sure_tasarrufu_yuzde"), 2),
                "tasarruf_std_sapma": round(std("sure_tasarrufu_yuzde"), 3),
                "ortalama_fitness": round(avg("fitness"), 6),
                "fitness_std_sapma": round(std("fitness"), 6),
                "ortalama_ilk_ceyrek_kritik_kapsam_yuzde": round(
                    avg("ilk_ceyrek_kritik_kapsam_yuzde"), 2
                ),
                "ortalama_algoritma_suresi_sn": round(avg("algoritma_suresi_sn"), 3),
                "minimum_kritik_kapsam_yuzde": min(r["kritik_kapsam_yuzde"] for r in group),
                "minimum_genel_kapsam_yuzde": min(r["genel_kapsam_yuzde"] for r in group),
                "toplam_on_kosul_ihlali": sum(r["kopuk_on_kosul"] for r in group),
            })

    summary_path = os.path.join(output_dir, "algoritma_ozet.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(summary_rows)

    columns = ["Paket", "Hedef", "Yontem", "Secilen", "Tasarruf", "Fitness", "Ilk %25 kritik"]
    table_data = [
        [
            str(row["test_sayisi"]), str(row["gereksinim_sayisi"]),
            row["algoritma"], f"{row['ortalama_secili_test']:.1f}",
            f"%{row['ortalama_tasarruf_yuzde']:.2f}",
            f"{row['ortalama_fitness']:.4f}",
            f"%{row['ortalama_ilk_ceyrek_kritik_kapsam_yuzde']:.1f}",
        ] for row in summary_rows
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis("off")
    ax.set_title("Telekom Test Secimi - Algoritmalarin Ozeti", pad=18, fontsize=15)
    table = ax.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1B263B")
            cell.set_text_props(color="white", weight="bold")
        else:
            group_num = (row_idx - 1) // len(algorithm_names)
            cell.set_facecolor("#EAF2F8" if group_num % 2 == 0 else "#FFFFFF")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "algoritma_ozet.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(SIZES))
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, name in enumerate(algorithm_names):
        means = [
            np.mean([r["sure_tasarrufu_yuzde"] for r in comparison_rows
                     if r["algoritma"] == name and r["test_sayisi"] == n])
            for n in SIZES
        ]
        ax.bar(x + (idx - 1.5) * 0.2, means, width=0.2, label=name)
    ax.set_xticks(x, [str(n) for n in SIZES])
    ax.set_xlabel("Test sayisi")
    ax.set_ylabel("Ortalama sure tasarrufu (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Greedy - GA - BMGO - IBMGO Karsilastirmasi")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "algoritma_karsilastirma.png"), dpi=200)
    plt.close(fig)

    # Her veri boyutunda ayni fitness tanimi vardir, ancak boyutlar arasi
    # mutlak fitness degerlerini basari siralamasi gibi okumamak gerekir.
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, name in enumerate(algorithm_names):
        means = [
            np.mean([r["fitness"] for r in comparison_rows
                     if r["algoritma"] == name and r["test_sayisi"] == n])
            for n in SIZES
        ]
        ax.bar(x + (idx - 1.5) * 0.2, means, width=0.2, label=name)
    ax.set_xticks(x, [str(n) for n in SIZES])
    ax.set_xlabel("Test sayisi")
    ax.set_ylabel("Ortalama fitness (dusuk daha iyi)")
    ax.set_title("Ayni Veri Boyutunda Fitness Karsilastirmasi")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "algoritma_fitness_karsilastirma.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in ("BMGO", "IBMGO", "GA"):
        curve = curves_1000[name]
        evaluations_per_step = {
            "BMGO": POP_SIZE * 4,
            "IBMGO": POP_SIZE * 5,
            "GA": POP_SIZE,
        }
        evaluations = POP_SIZE + np.arange(1, len(curve) + 1) * evaluations_per_step[name]
        ax.plot(evaluations, curve, label=name)
    ax.set_xlabel("Degerlendirilen aday sayisi")
    ax.set_ylabel("En iyi fitness (dusuk daha iyi)")
    ax.set_title("1000 Testte Es Aday Butcesiyle Yakinsama")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bmgo_iyilestirme_yakinsama.png"), dpi=200)
    plt.close(fig)

    print(f"\nKarsilastirma CSV: {comparison_path} | Ozet: {summary_path}")
    print("Karsilastirma grafikleri: algoritma_karsilastirma.png, "
          "algoritma_fitness_karsilastirma.png, bmgo_iyilestirme_yakinsama.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMGO telekom regresyon deneyleri")
    parser.add_argument("--runs", type=int, default=3, help="Stokastik algoritma tekrar sayisi")
    parser.add_argument("--data-seed", type=int, default=42, help="Veri uretim tohumu")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Sonuc dosyalari dizini; onceki deneyleri koruyabilirsiniz")
    parser.add_argument(
        "--profile", choices=("standard", "expanded"), default="expanded",
        help="Telekom veri kumesi zorlugu",
    )
    args = parser.parse_args()
    run_experiment(num_runs=args.runs, scenario_profile=args.profile,
                   data_seed=args.data_seed, output_dir=args.output_dir)
