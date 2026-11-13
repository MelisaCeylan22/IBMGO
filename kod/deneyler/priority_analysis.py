"""Ön koşullara saygılı kritik kapsam önceliklendirme deneyi.

BMGO'nun test seçme aşaması değiştirilmez. Seçilen aynı test kümesinin
akıllı sırası, mümkün olan rastgele bağımlılık-geçerli sıralarla kıyaslanır.
Hata tespit matrisi olmadığından bu analiz APFD değildir.

Çalıştırma: python -m kod.deneyler.priority_analysis --random-orders 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kod.veri.data_generator import generate_telecom_data


SIZES = (50, 100, 250, 500, 1000)


def _read_order(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file, delimiter=";"))
    rows.sort(key=lambda row: int(row["oncelik_sirasi"]))
    return np.asarray([int(row["id"].split("_")[-1]) - 1 for row in rows], dtype=int)


def _random_valid_order(selected: np.ndarray, dependencies: np.ndarray, rng):
    """Sadece seçilen testler üzerinde rastgele topolojik sıralama."""
    remaining = set(map(int, selected))
    ordered = []
    while remaining:
        ready = sorted(
            child for child in remaining
            if not any(dependencies[parent, child] for parent in remaining)
        )
        if not ready:
            raise ValueError("Test on kosullarinda dongu veya veri hatasi")
        chosen = int(rng.choice(ready))
        ordered.append(chosen)
        remaining.remove(chosen)
    return np.asarray(ordered, dtype=int)


def _critical_metrics(order, coverage, durations, critical, grid=None):
    """Biten testlere göre biriken kritik kapsam; hata oranı değildir."""
    hits = coverage[order][:, critical] != 0
    after_each = 100.0 * np.count_nonzero(
        np.maximum.accumulate(hits, axis=0), axis=1
    ) / len(critical)
    elapsed = np.cumsum(durations[order]).astype(float)
    frac_time = elapsed / elapsed[-1]
    first_quarter_count = max(1, int(np.ceil(len(order) / 4)))
    first_quarter = float(after_each[first_quarter_count - 1])
    ended_by_quarter = int(np.searchsorted(frac_time, 0.25, side="right"))
    quarter_time = float(after_each[ended_by_quarter - 1]) if ended_by_quarter else 0.0
    before_each = np.r_[0.0, after_each[:-1]] / 100.0
    # Zaman boyunca basamaklı fonksiyonun gerçek alanı; test yarım
    # çalışmışken kapsam artmış gibi lineer enterpolasyon yapmayız.
    auc = float(np.sum(before_each * durations[order]) / elapsed[-1])
    if grid is None:
        return first_quarter, quarter_time, auc, None
    finished = np.searchsorted(frac_time, grid, side="right")
    values = np.r_[0.0, after_each]
    return first_quarter, quarter_time, auc, values[finished]


def run(random_orders: int = 100, data_seed: int = 42,
        output_dir: str = "sonuclar/nihai_bmgo"):
    if random_orders < 1:
        raise ValueError("En az bir rastgele kontrol sirasi gerekir")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    grid = np.linspace(0, 1, 201)
    rows = []
    plot_data = None

    for size in SIZES:
        coverage, durations, _, _, deps, meta = generate_telecom_data(
            size, seed=data_seed, include_metadata=True, scenario_profile="expanded"
        )
        critical = np.asarray(meta["critical_feature_indices"], dtype=int)
        for alg, filename in (
            ("BMGO", f"telekom_{size}_bmgo_ilk_secili_testler.csv"),
            ("IBMGO", f"telekom_{size}_secili_testler.csv"),
        ):
            order = _read_order(output / filename)
            if len(set(map(int, order))) != len(order):
                raise ValueError(f"{alg}/{size}: tekrarlanan test kimligi")
            # Sıralama, bütün ebeveynlerinin konumundan sonra gelmeli.
            positions = {int(idx): pos for pos, idx in enumerate(order)}
            for child in order:
                parents = np.flatnonzero(deps[:, child])
                if any(int(p) not in positions or positions[int(p)] >= positions[int(child)]
                       for p in parents):
                    raise ValueError(f"{alg}/{size}: gecersiz on kosul sirasi")

            a, b, auc, curve = _critical_metrics(order, coverage, durations, critical, grid)
            controls = np.asarray([
                _critical_metrics(
                    _random_valid_order(order, deps, rng), coverage,
                    durations, critical, grid
                )[:3]
                for _ in range(random_orders)
            ])
            rows.append({
                "test_sayisi": size,
                "algoritma": alg,
                "secili_test": len(order),
                "kritik_kural_sayisi": len(critical),
                "ilk_yuzde25_testte_kritik_kapsam": round(a, 2),
                "rastgele_ilk_yuzde25_test_ortalama": round(float(controls[:, 0].mean()), 2),
                "ilk_yuzde25_surede_kritik_kapsam": round(b, 2),
                "rastgele_ilk_yuzde25_sure_ortalama": round(float(controls[:, 1].mean()), 2),
                "kritik_kapsam_auc": round(auc, 4),
                "rastgele_auc_ortalama": round(float(controls[:, 2].mean()), 4),
                "rastgele_tekrar": random_orders,
            })
            if size == 1000 and alg == "IBMGO":
                curves = np.asarray([
                    _critical_metrics(
                        _random_valid_order(order, deps, rng),
                        coverage, durations, critical, grid,
                    )[3]
                    for _ in range(random_orders)
                ])
                plot_data = (grid, curve, curves.mean(axis=0),
                             np.percentile(curves, 10, axis=0),
                             np.percentile(curves, 90, axis=0))

            print(f"{size:4} | {alg:5} | %25 sure kritik kapsam: "
                  f"%{b:.1f} (rastgele %{controls[:, 1].mean():.1f}) | "
                  f"AUC {auc:.3f} (rastgele {controls[:, 2].mean():.3f})")

    csv_path = output / "onceliklendirme_analizi.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    if plot_data is not None:
        x, smart, random_avg, random_lo, random_hi = plot_data
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.step(x * 100, smart, where="post", label="IBMGO secimi + oncelikli sira", linewidth=2.2)
        ax.plot(x * 100, random_avg, label="Ayni testlerde rastgele gecerli sira", linewidth=2)
        ax.fill_between(x * 100, random_lo, random_hi, alpha=0.15,
                        label="Rastgele %10-%90 araligi")
        ax.axvline(25, linestyle="--", color="gray", alpha=0.65)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Secili testlerin toplam kosum suresinin tamamlanan yuzdesi")
        ax.set_ylabel("Tamamlanan kritik gereksinim kapsami (%)")
        ax.set_title("1000 Testte Kritik Kapsama Erisim Hizi (APFD Degil)")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output / "onceliklendirme_kritik_kapsam.png", dpi=200)
        plt.close(fig)

    print(f"\nAnaliz: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-orders", type=int, default=100)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="sonuclar/nihai_bmgo")
    options = parser.parse_args()
    run(options.random_orders, options.data_seed, options.output_dir)
