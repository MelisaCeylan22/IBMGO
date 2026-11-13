"""IBMGO dayanıklılık analizi: sabit hiperparametre, çoklu veri/algoritma seed.

Örnek:
  python -m kod.deneyler.robustness_analysis --inputs sonuclar/dayaniklilik_83/ablasyon_veri83.csv ...

Önemli: Greedy deterministiktir ve her veri kümesi için TEK sonuç verir.
IBMGO/GA eşleştirmesi aynı veri kümesi ve optimizer seed üzerinden yapılır.
Üç veri kümesi, üç gerçek dünya sistemi anlamına gelmez.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "BMGO_boyuta_adaptif": "IBMGO",
    "GA": "GA",
    "Greedy": "Greedy",
}
SIZES = (50, 100, 250, 500, 1000)
FIELDS = ("fitness", "sure_tasarrufu", "secilen_test", "genel_kapsam", "algoritma_suresi")


def _read_csv(paths: list[Path]):
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                if row["algoritma"] not in DISPLAY:
                    raise ValueError(f"Beklenmeyen algoritma: {row['algoritma']}")
                for key in ("test_sayisi", "veri_seed", "kritik_eksik", "kopuk_on_kosul"):
                    row[key] = int(row[key])
                for key in FIELDS:
                    row[key] = float(row[key])
                if row["algoritma"] == "Greedy":
                    row["algoritma_seed"] = None
                else:
                    row["algoritma_seed"] = int(row["algoritma_seed"])
                    if int(row["aday_degerlendirmesi"]) != 1212:
                        raise ValueError("Eşit 1212 aday değerlendirme bütçesi bozulmuş")
                if row["kritik_eksik"] or row["kopuk_on_kosul"] or row["genel_kapsam"] < 89.995:
                    raise ValueError(f"Geçersiz test paketi: {row}")
                rows.append(row)
    return rows


def _validate_design(rows, runs: int, sizes: tuple[int, ...]):
    by_key = defaultdict(list)
    seeds = set()
    for row in rows:
        k = (row["test_sayisi"], row["veri_seed"], row["algoritma"])
        by_key[k].append(row)
        seeds.add(row["veri_seed"])
    for size in sizes:
        for data_seed in seeds:
            for algorithm in DISPLAY:
                arr = by_key[(size, data_seed, algorithm)]
                expected = 1 if algorithm == "Greedy" else runs
                if len(arr) != expected:
                    raise ValueError(f"{size}/{data_seed}/{algorithm}: beklenen {expected}, var {len(arr)}")
                if algorithm != "Greedy":
                    got = {r["algoritma_seed"] for r in arr}
                    if got != set(range(42, 42 + runs)):
                        raise ValueError(f"{size}/{data_seed}/{algorithm} seed kümesi hatalı: {got}")
    if len(rows) != len(sizes) * len(seeds) * (2 * runs + 1):
        raise ValueError("Mükerrer veya eksik koşum var")
    return sorted(seeds)


def _summary(rows, sizes, data_seeds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["test_sayisi"], row["algoritma"])].append(row)
    summaries = []
    for size in sizes:
        for alg, display in DISPLAY.items():
            arr = grouped[(size, alg)]
            s = {"test_sayisi": size, "algoritma": display, "kayıt": len(arr),
                 "veri_kümesi": len(data_seeds)}
            for f in FIELDS:
                vals = [r[f] for r in arr]
                s[f + "_ort"] = round(mean(vals), 6)
                s[f + "_ss"] = round(stdev(vals), 6) if len(vals) > 1 else 0.0
            summaries.append(s)
    return summaries


def _hierarchical_bootstrap(per_dataset, rng, n_boot=6000):
    """Önce veri kümesi, sonra aynı kümede eşleşmiş koşumları örnekle."""
    groups = list(per_dataset.values())
    count = len(groups)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        indices = rng.integers(0, count, count)
        means = []
        for idx in indices:
            sample = groups[idx]
            means.append(np.mean(rng.choice(sample, len(sample), replace=True)))
        samples[b] = np.mean(means)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def _paired(rows, sizes, seeds, rng):
    indexed = {
        (r["test_sayisi"], r["veri_seed"], r["algoritma_seed"], r["algoritma"]): r
        for r in rows if r["algoritma"] != "Greedy"
    }
    pairs = []
    details = []
    for size in sizes:
        by_seed = defaultdict(list)
        wins = ties = losses = 0
        for ds in seeds:
            for os in range(42, 42 + len([r for r in rows if r["test_sayisi"] == size
                                             and r["veri_seed"] == ds
                                             and r["algoritma"] == "GA"])):
                bmgo = indexed[(size, ds, os, "BMGO_boyuta_adaptif")]
                ga = indexed[(size, ds, os, "GA")]
                delta = bmgo["fitness"] - ga["fitness"]
                by_seed[ds].append(delta)
                if delta < -0.000001:
                    wins += 1
                elif delta > 0.000001:
                    losses += 1
                else:
                    ties += 1
                details.append({"test_sayisi": size, "veri_seed": ds, "algoritma_seed": os,
                                "bmgo_fitness": bmgo["fitness"], "ga_fitness": ga["fitness"],
                                "fark_bmgo_eksi_ga": round(delta, 6),
                                "bmgo_tasarruf": bmgo["sure_tasarrufu"],
                                "ga_tasarruf": ga["sure_tasarrufu"]})
        diffs = [v for vv in by_seed.values() for v in vv]
        ci_lo, ci_hi = _hierarchical_bootstrap(by_seed, rng)
        pairs.append({"test_sayisi": size, "çift": len(diffs),
                      "bmgo_daha_iyi": wins, "eşit": ties, "ga_daha_iyi": losses,
                      "bmgo_kazanma_yüzdesi": round(100 * wins / len(diffs), 1),
                      "ortalama_fitness_farkı": round(mean(diffs), 6),
                      "bootstrap_alt_95": round(float(ci_lo), 6),
                      "bootstrap_üst_95": round(float(ci_hi), 6)})
    return pairs, details


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _plot(summaries, pairs, details, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    by_alg = defaultdict(list)
    for row in summaries:
        by_alg[row["algoritma"]].append(row)
    colors = {"IBMGO": "#1169a0", "GA": "#e57d24", "Greedy": "#6e7d93"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for alg in ("IBMGO", "GA", "Greedy"):
        arr = by_alg[alg]
        xs = [str(r["test_sayisi"]) for r in arr]
        ax1.plot(xs, [r["fitness_ort"] for r in arr], marker="o", lw=2, label=alg, color=colors[alg])
        ax2.plot(xs, [r["sure_tasarrufu_ort"] for r in arr], marker="o", lw=2,
                 label=alg, color=colors[alg])
    ax1.set(title="Fitness (düşük daha iyi)", xlabel="Başlangıç test sayısı", ylabel="Ortalama fitness")
    ax2.set(title="Test koşum süresi tasarrufu (yüksek daha iyi)",
            xlabel="Başlangıç test sayısı", ylabel="Ortalama tasarruf (%)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
    fig.suptitle("IBMGO / Genetik Algoritma / Greedy | 3 veri kümesi", fontsize=14)
    fig.savefig(output / "01_ana_karsilastirma.png", dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    labels = [str(p["test_sayisi"]) for p in pairs]
    x = np.arange(len(pairs))
    bottom = np.zeros(len(pairs))
    total = np.array([p["çift"] for p in pairs])
    for key, label, color in (("bmgo_daha_iyi", "IBMGO daha iyi", "#1169a0"),
                              ("eşit", "Eşit", "#b9bec8"),
                              ("ga_daha_iyi", "GA daha iyi", "#e57d24")):
        values = 100 * np.array([p[key] for p in pairs]) / total
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        for xi, b, v in zip(x, bottom, values):
            if v >= 12:
                ax.text(xi, b + v / 2, f"%{v:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if key != "eşit" else "black")
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set(title="Aynı test verisi ve algoritma seed'inde IBMGO / GA karşılaştırması",
           xlabel="Test sayısı", ylabel="Eşleşmiş koşumların oranı (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.savefig(output / "02_eslesmis_kazanma.png", dpi=200, facecolor="white")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, layout="constrained")
    seen = sorted({r["veri_seed"] for r in details})
    for ax, size in zip(axs, (250, 500, 1000)):
        values = [mean([r["fark_bmgo_eksi_ga"] for r in details
                        if r["test_sayisi"] == size and r["veri_seed"] == ds]) for ds in seen]
        bars = ax.bar([str(s) for s in seen], values,
                      color=["#1169a0" if v < 0 else "#e57d24" for v in values])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{size} test")
        ax.set_xlabel("Veri kümesi seed")
        ax.grid(axis="y", alpha=0.2)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:+.4f}", (bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, 4 if v >= 0 else -13), textcoords="offset points",
                        ha="center", fontsize=9)
    axs[0].set_ylabel("Ortalama fitness farkı (IBMGO − GA)")
    fig.suptitle("Veri kümesine göre karşılaştırma | Negatif IBMGO lehine", fontsize=13)
    fig.savefig(output / "03_veri_kumesi_duyarliligi.png", dpi=200, facecolor="white")
    plt.close(fig)


def _report(path, summaries, pairs, seeds, runs):
    group = {(x["test_sayisi"], x["algoritma"]): x for x in summaries}
    out = ["# IBMGO sağlamlık ve tekrar edilebilirlik analizi", "",
           "**Deney tasarımı (önceden sabitlendi):** IBMGO'nin boyuta uyarlanan ayarları "
           "değiştirilmeden test edilmiştir. GA da 1212 fitness değerlendirmesi "
           "kullanır. Greedy deterministiktir; aynı veri kümesinde tekrar çalıştırılmamıştır.", "",
           f"- Sentetik telekom veri seti başlangıçları: **{', '.join(map(str, seeds))}** ({len(seeds)} farklı set).",
           f"- Her set/test büyüklüğünde IBMGO ve GA için **{runs}** ayrı optimizer başlangıcı.",
           f"- Her test büyüklüğünde **{len(seeds)*runs}** eşleşmiş IBMGO/GA deneyi; "
           f"Greedy için **{len(seeds)}** ayrı veri kümesi sonucu.",
           "- Kritik gereksinim kapsamı %100, genel gereksinim kapsamı en az %90, "
           "ön koşul ihlali sıfır — tüm kayıtlarda doğrulandı.",
           "- Değerler sentetiktir. Gerçek hata tespit matrisi olmadığı için APFD hesaplanmamıştır.", "",
           "## Ana tablo", "",
           "| Test | IBMGO fitness | GA fitness | Greedy fitness | IBMGO tasarruf | GA tasarruf | Greedy tasarruf |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for size in SIZES:
        b, g, gr = (group[(size, a)] for a in ("IBMGO", "GA", "Greedy"))
        out.append(f"| {size} | {b['fitness_ort']:.4f} ± {b['fitness_ss']:.4f} | "
                   f"{g['fitness_ort']:.4f} ± {g['fitness_ss']:.4f} | "
                   f"{gr['fitness_ort']:.4f} | %{b['sure_tasarrufu_ort']:.2f} | "
                   f"%{g['sure_tasarrufu_ort']:.2f} | %{gr['sure_tasarrufu_ort']:.2f} |")
    out += ["", "*± işareti IBMGO ve GA için 30 optimizer/veri koşumunun örnek standart sapmasıdır. "
            "Greedy ortalaması yalnız üç veri kümesi üzerindendir.*", "",
            "## IBMGO − GA eşleşmiş farkları", "",
            "Fitness düşük daha iyi; **negatif fark IBMGO lehine**. "
            "Güven aralıkları veri kümesi ve optimizer başlangıcı düzeyinde "
            "hiyerarşik bootstrap ile hesaplanan betimleyici aralıklardır; yalnız "
            "üç veri kümesi bulunduğundan kesin genelleme veya anlamlılık "
            "kanıtı olarak yorumlanmamalıdır.", "",
            "| Test | IBMGO kazanma | Eşit | GA kazanma | Ort. fark | %95 bootstrap aralığı |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for p in pairs:
        out.append(f"| {p['test_sayisi']} | {p['bmgo_daha_iyi']} | {p['eşit']} | "
                   f"{p['ga_daha_iyi']} | {p['ortalama_fitness_farkı']:+.5f} | "
                   f"[{p['bootstrap_alt_95']:+.5f}, {p['bootstrap_üst_95']:+.5f}] |")
    out += ["", "## Sunumda güvenle söylenebilecekler", "",
            "- IBMGO, bu sentetik telekom probleminde her tekrarda geçerli ve "
            "kısıtlara uyan test paketi oluşturmuştur.",
            "- Farklı başlangıçlarda sonuçların ne kadar değiştiği ve "
            "hangi veri kümelerinde hangi yöntemin öne geçtiği artık "
            "raporlanabilir; tek bir iyi koşuma dayanmıyoruz.",
            "- Greedy'nin gerçek çalışma zamanı ile IBMGO/GA'nın çalışma zamanı "
            "aynı donanımda ölçülse bile tek geçişli Greedy'nin aday bütçesi eşit "
            "değildir: kalite ve hesaplama maliyeti farklı eksenlerde yorumlanmalıdır.",
            "- Eşleşmiş optimizer seed'leri ortak veri koşullarını kontrol eder, "
            "ancak IBMGO ve GA'nın aynı rassal sayı akışını kullandığı anlamına gelmez.", "",
            "## IBMGO'nin çalışma mantığı (mekanizma açıklaması)", "",
            "- 50–250 test: farklı başlangıç çözümleri, ikili dönüşüm ve "
            "yerel komşuluk araması birlikte kullanılır.",
            "- 500 test: referans MGO hareketleri, kaotik başlangıç, doğrusal "
            "olmayan kontrol ve daha geniş komşuluk araması devrededir.",
            "- 1000 test: referans ve uyarlanmış hareketler karıştırılır; "
            "komşuluk araması daha küçük değişikliklere döner.",
            "- Amaç, geniş aday uzayını araştırırken seçilmiş iyi test "
            "paketlerinin çevresinde de arama yapmaktır. Bu bir tasarım "
            "gerekçesidir; tek başına hangi bileşenin fayda sağladığını "
            "kanıtlamaz. Nedensel iddia için ayrı ablasyon sonuçları gerekir.", "",
            "## Sınırlılıklar", "",
            "- Yalnız üç sentetik veri kümesi: gerçek operatör üretim verisine "
            "genellenemez. Verinin üretim kuralları optimize edilmemiştir.",
            "- 1212 fitness değerlendirmesi eşit olsa da algoritmaların duvar saati "
            "süresi ve aday başına maliyeti farklıdır.",
            "- Farklı test büyüklükleri farklı gereksinim ve test havuzlarıdır; "
            "fitness değerlerini *boyutlar arasında* doğrudan sıralamak yerine "
            "her boyut içinde algoritmalar arası karşılaştırma yapılmalıdır.",
            "- Test süreleri toplamı optimizasyon hedefidir; CI/CD boru hattındaki "
            "gerçek yürütme süresi sahada ölçülmemiştir.",
            "- Hata erken yakalama performansı için doğrulanmış arıza-test matrisi gerekir; "
            "önceliklendirme göstergeleri gerçek APFD yerine geçmez.", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", default="sonuclar/dayaniklilik_analizi", type=Path)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    rows = _read_csv(args.inputs)
    seeds = _validate_design(rows, args.runs, SIZES)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summaries = _summary(rows, SIZES, seeds)
    pairs, details = _paired(rows, SIZES, seeds, np.random.default_rng(20261119))
    _write_csv(output / "algoritma_ozet.csv", summaries)
    _write_csv(output / "eslesmis_karsilastirma.csv", pairs)
    _write_csv(output / "eslesmis_ham_farklar.csv", details)
    (output / "analiz_verileri.json").write_text(
        json.dumps({"veri_seedleri": seeds, "tekrar": args.runs,
                    "ozet": summaries, "eslesmis": pairs,
                    "eslesmis_ham": details, "ham_kosumlar": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(summaries, pairs, details, output)
    _report(output / "DAYANIKLILIK_RAPORU.md", summaries, pairs, seeds, args.runs)
    print(f"Doğrulandı: {len(seeds)} veri kümesi, {args.runs} optimizer tekrarı, "
          f"{len(rows)} toplam algoritma kaydı.")
    for p in pairs:
        print(f"{p['test_sayisi']:4} test | IBMGO/GA eşleşmesi "
              f"{p['bmgo_daha_iyi']}/{p['eşit']}/{p['ga_daha_iyi']} "
              f"(kazanma/eşit/kaybetme) | fitness farkı "
              f"{p['ortalama_fitness_farkı']:+.5f}")
    print(f"Rapor: {output / 'DAYANIKLILIK_RAPORU.md'}")


if __name__ == "__main__":
    main()
