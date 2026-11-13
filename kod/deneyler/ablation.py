"""BMGO hareket / baslangic / kontrol / spiral etkisini ayri ayri olcer.

Ornek: python -m kod.deneyler.ablation --sizes 250 500 --runs 2

Butun populasyon yontemleri 1212 fitness degerlendirmesine sahiptir.
Degisiklikler tek tek acilir; gercek hata tespit iddiasi yoktur.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kod.algoritmalar.baselines import genetic_algorithm_optimizer, greedy_regression_selector
from kod.algoritmalar.bmgo import binary_mountain_gazelle_optimizer, dimension_adaptive_settings
from kod.veri.data_generator import generate_telecom_data
from kod.algoritmalar.fitness import get_suite_metrics


POP = 12
BUDGET = POP * (1 + 4 * 25)
BASE = dict(pop_size=POP, w_time=0.4, w_count=0.4, w_redundancy=0.2)
PLUS = dict(max_iter=20, adaptive_binary=True,
            diverse_initialization=True, local_search_trials=12)
VARIANTS = {
    "BMGO_ilk": dict(max_iter=25),
    "BMGO_artı": PLUS.copy(),
    "Referans_hareket": dict(PLUS, reference_mgo_formulas=True),
    "Artı_kaotik": dict(PLUS, reference_mgo_formulas=True,
                        chaotic_initialization=True),
    "Artı_dogrusal_olmayan": dict(PLUS, reference_mgo_formulas=True,
                                 chaotic_initialization=True, nonlinear_control=True),
    "Karma_hareket_25": dict(PLUS, reference_mgo_formulas=True,
                            reference_mgo_probability=0.25,
                            chaotic_initialization=True, nonlinear_control=True),
    "Karma_hareket_50": dict(PLUS, reference_mgo_formulas=True,
                            reference_mgo_probability=0.50,
                            chaotic_initialization=True, nonlinear_control=True),
    "Karma_hareket_75": dict(PLUS, reference_mgo_formulas=True,
                            reference_mgo_probability=0.75,
                            chaotic_initialization=True, nonlinear_control=True),
    "Ince_flip_1": dict(PLUS, reference_mgo_formulas=True,
                       chaotic_initialization=True, nonlinear_control=True,
                       flip_rate_start=0.02, flip_rate_end=0.01),
    "Ince_flip_2": dict(PLUS, reference_mgo_formulas=True,
                       chaotic_initialization=True, nonlinear_control=True,
                       flip_rate_start=0.012, flip_rate_end=0.006),
    "Genis_komsu": dict(PLUS, reference_mgo_formulas=True,
                        chaotic_initialization=True, nonlinear_control=True,
                        local_search_strategy="large"),
    "Genis_komsu_eski": dict(PLUS, local_search_strategy="large"),
    "BMGO_boyuta_adaptif": PLUS.copy(),  # asil ayarlar dimension_adaptive_settings
    "Artı_sigmoid": dict(PLUS, reference_mgo_formulas=True,
                        chaotic_initialization=True, nonlinear_control=True,
                        binary_transfer="s_assign"),
    "Artı_spiral": dict(PLUS, reference_mgo_formulas=True,
                       chaotic_initialization=True, nonlinear_control=True,
                       local_search_trials=6, spiral_search_trials=6),
}


def _check_budget():
    for name, config in VARIANTS.items():
        evaluations = POP * (1 + config["max_iter"] * (
            4 * POP + config.get("local_search_trials", 0)
            + config.get("spiral_search_trials", 0)
        ) / POP)
        if int(evaluations) != BUDGET:
            raise AssertionError(f"{name} aday degerlendirme butcesi es degil")


def run(sizes=(250, 500), runs=2, data_seed=42, output_dir="sonuclar",
        variant_names=None):
    if runs < 1 or not sizes:
        raise ValueError("Kosum sayisi ve test boyutu pozitif olmali")
    _check_budget()
    variants = VARIANTS if variant_names is None else {
        name: VARIANTS[name] for name in variant_names
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in sizes:
        coverage, durations, _, _, deps, meta = generate_telecom_data(
            size, seed=data_seed, include_metadata=True, scenario_profile="expanded"
        )
        critical = meta["critical_feature_indices"]
        common = dict(coverage_matrix=coverage, durations=durations,
                      dependency_matrix=deps, critical_feature_indices=critical,
                      min_coverage=0.90)
        for run_index in range(runs):
            seed = 42 + run_index
            for name, config in variants.items():
                effective_config = (
                    dict(max_iter=20, **dimension_adaptive_settings(size))
                    if name == "BMGO_boyuta_adaptif" else config
                )
                start = perf_counter()
                solution, score, _ = binary_mountain_gazelle_optimizer(
                    **common, **BASE, **effective_config,
                    random_seed=seed,
                )
                elapsed = perf_counter() - start
                metric = get_suite_metrics(solution, coverage, durations, deps, critical)
                if (metric["critical_missing"] != 0 or
                        metric["broken_dependencies"] != 0 or
                        metric["coverage_ratio"] < 0.90 - 1e-9):
                    raise AssertionError(f"{name} gecersiz cozum")
                rows.append({
                    "test_sayisi": size, "veri_seed": data_seed, "algoritma": name,
                    "algoritma_seed": seed, "fitness": round(score, 6),
                    "secilen_test": metric["selected_count"],
                    "sure_tasarrufu": round(100 * (1 - metric["time_ratio"]), 2),
                    "genel_kapsam": round(100 * metric["coverage_ratio"], 2),
                    "kritik_eksik": metric["critical_missing"],
                    "kopuk_on_kosul": metric["broken_dependencies"],
                    "aday_degerlendirmesi": BUDGET,
                    "algoritma_suresi": round(elapsed, 3),
                })
                print(f"{size:4} | seed={seed} | {name:23} | "
                      f"fitness={score:.5f} | tasarruf=%{rows[-1]['sure_tasarrufu']:.2f}",
                      flush=True)

            start = perf_counter()
            sol, score, _ = genetic_algorithm_optimizer(
                **common, pop_size=POP, max_generations=100,
                random_seed=seed,
            )
            elapsed = perf_counter() - start
            metric = get_suite_metrics(sol, coverage, durations, deps, critical)
            rows.append({
                "test_sayisi": size, "veri_seed": data_seed, "algoritma": "GA",
                "algoritma_seed": seed, "fitness": round(score, 6),
                "secilen_test": metric["selected_count"],
                "sure_tasarrufu": round(100 * (1 - metric["time_ratio"]), 2),
                "genel_kapsam": round(100 * metric["coverage_ratio"], 2),
                "kritik_eksik": metric["critical_missing"],
                "kopuk_on_kosul": metric["broken_dependencies"],
                "aday_degerlendirmesi": BUDGET,
                "algoritma_suresi": round(elapsed, 3),
            })

        sol, score = greedy_regression_selector(**common)
        metric = get_suite_metrics(sol, coverage, durations, deps, critical)
        rows.append({
            "test_sayisi": size, "veri_seed": data_seed, "algoritma": "Greedy",
            "algoritma_seed": "tek", "fitness": round(score, 6),
            "secilen_test": metric["selected_count"],
            "sure_tasarrufu": round(100 * (1 - metric["time_ratio"]), 2),
            "genel_kapsam": round(100 * metric["coverage_ratio"], 2),
            "kritik_eksik": metric["critical_missing"],
            "kopuk_on_kosul": metric["broken_dependencies"],
            "aday_degerlendirmesi": "tek gecis",
            "algoritma_suresi": 0.0,
        })

    path = output / f"ablasyon_veri{data_seed}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algoritma"], row["test_sayisi"])].append(row["fitness"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in (*variants, "GA", "Greedy"):
        values = [np.mean(grouped[(name, s)]) for s in sizes]
        ax.plot(sizes, values, marker="o", label=name)
    ax.set_xticks(sizes)
    ax.set_xlabel("Baslangic test sayisi")
    ax.set_ylabel("Ortalama fitness (dusuk daha iyi)")
    ax.set_title(f"BMGO Ablasyon ve GA Karsilastirmasi (veri seed={data_seed})")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / f"ablasyon_veri{data_seed}.png", dpi=190)
    plt.close(fig)
    print(f"\nSonuclar: {path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 500])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="sonuclar")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS))
    args = parser.parse_args()
    run(args.sizes, args.runs, args.data_seed, args.output_dir, args.variants)
