"""Kapsam denetimleri, yöntem özetleri ve eşleştirilmiş deney grafikleri."""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from kod.deneyler.notebook_deneyi import YONTEMLER


RENKLER = {"IBMGO": "#087F8C", "GA": "#8B55B8", "Greedy": "#C05C24"}
MUREKKEP = "#17283F"


def deney_sonuclarini_goster(sonuclar):
    ozet = sonuclar.groupby(["Havuz", "Test sayısı", "Yöntem"]).agg(**{
        "Koşum": ("Fitness", "size"),
        "Ort. puan": ("Fitness", "mean"),
        "Ort. seçilen test": ("Seçilen test", "mean"),
        "Ort. paket (dk)": ("Seçili paket (dk)", "mean"),
        "Ort. tasarruf (%)": ("Tasarruf (%)", "mean"),
        "En düşük genel kapsam (%)": ("Genel kapsam (%)", "min"),
        "En düşük kritik kapsam (%)": ("Kritik kapsam (%)", "min"),
        "Ön koşul ihlali": ("Ön koşul ihlali", "max"),
        "Ort. arama (sn)": ("Paket seçme (sn)", "mean"),
    })
    display(ozet.round({
        "Ort. puan": 5, "Ort. seçilen test": 1,
        "Ort. paket (dk)": 1, "Ort. tasarruf (%)": 2,
        "En düşük genel kapsam (%)": 2, "En düşük kritik kapsam (%)": 0,
        "Ort. arama (sn)": 2,
    }))
    for havuz, veri in sonuclar.groupby("Havuz", sort=False):
        _havuz_grafigini_goster(veri, havuz)
    if set(sonuclar["Havuz"]) == {"A", "B"}:
        _seyreltme_etkisini_goster(sonuclar)


def _havuz_grafigini_goster(veri, havuz):
    boyutlar = sorted(veri["Test sayısı"].unique())
    y = np.arange(len(boyutlar))
    fig, (sure, secim, eslesme) = plt.subplots(
        1, 3, figsize=(15, max(4.6, 2.6 + 0.6 * len(boyutlar))),
        sharey=True, layout="constrained", facecolor="white",
    )
    ad = {"A": "Orijinal", "B": "Seyrek kritik kapsam"}[havuz]
    fig.suptitle(f"Havuz {havuz} · {ad}   |   {len(veri)} koşum",
                 fontsize=17, fontweight="bold", color=MUREKKEP)

    for i, yontem in enumerate(YONTEMLER):
        renk = RENKLER[yontem]
        yontem_verisi = veri[veri["Yöntem"] == yontem]
        kayma = (i - 1) * 0.21
        for eksen, kolon in ((sure, "Tasarruf (%)"), (secim, "Seçilen test")):
            for konum, n in zip(y, boyutlar):
                degerler = yontem_verisi.loc[yontem_verisi["Test sayısı"] == n, kolon]
                eksen.scatter(degerler, np.full(len(degerler), konum + kayma),
                              s=19, color=renk, alpha=0.3, linewidths=0, zorder=2)
                eksen.scatter(degerler.mean(), konum + kayma, s=48, marker="D",
                              color=renk, edgecolor="white", linewidth=0.6, zorder=3,
                              label=yontem if konum == 0 else None)

    eslesmis = veri[veri["Yöntem"].isin(["IBMGO", "GA"])].pivot(
        index=["Test sayısı", "Veri tohumu", "Algoritma tohumu"], columns="Yöntem", values="Fitness",
    )
    farklar = eslesmis["GA"] - eslesmis["IBMGO"]
    eslesme.axvline(0, color="#52667D", lw=1, ls="--")
    for konum, n in zip(y, boyutlar):
        fark = farklar.loc[n]
        eslesme.scatter(fark, np.full(len(fark), konum), s=23,
                        color=np.where(fark >= 0, RENKLER["IBMGO"], RENKLER["GA"]), alpha=0.45)
        eslesme.scatter(fark.mean(), konum, s=52, marker="D", color=MUREKKEP,
                        edgecolor="white", linewidth=0.6, zorder=3)
    genislik = max(float(farklar.abs().max()) * 1.18, 0.0005)
    eslesme.set_xlim(-genislik, genislik)
    eslesme.ticklabel_format(axis="x", style="plain", useOffset=False)

    sure.set(title="1. Ne kadar süre kazanıldı?", xlabel="Tam pakete göre tahmini tasarruf (%)")
    secim.set(title="2. Kaç test seçildi?", xlabel="Seçili test sayısı")
    eslesme.set(title="3. Eşleşen koşumlarda puan farkı", xlabel="GA − IBMGO · sağ taraf IBMGO lehine")
    sure.set_yticks(y, [f"{n:,} test".replace(",", ".") for n in boyutlar])
    sure.set_ylim(len(boyutlar) - 0.55, -0.45)
    for eksen in (sure, secim, eslesme):
        eksen.set_axisbelow(True)
        eksen.grid(axis="x", color="#E4EAF2")
        eksen.tick_params(length=0, labelcolor=MUREKKEP)
        eksen.spines[["top", "right", "left"]].set_visible(False)
        eksen.title.set_color(MUREKKEP)
        eksen.title.set_weight("semibold")
    sure.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False)
    secim.text(0.5, -0.31, "Nokta: koşum   ◆  Ortalama", transform=secim.transAxes,
               ha="center", color="#52667D", fontsize=9)
    plt.show()


def _seyreltme_etkisini_goster(sonuclar):
    anahtar = ["Test sayısı", "Veri tohumu", "Algoritma tohumu", "Yöntem"]
    a = sonuclar[sonuclar["Havuz"] == "A"].set_index(anahtar)
    b = sonuclar[sonuclar["Havuz"] == "B"].set_index(anahtar)
    fark = (b["Seçili paket (dk)"] - a["Seçili paket (dk)"]).rename("B − A")
    ozet = fark.groupby(level=["Test sayısı", "Yöntem"]).mean().unstack().reindex(columns=list(YONTEMLER))
    eksen = ozet.plot.bar(figsize=(9.5, 4.5), color=[RENKLER[y] for y in YONTEMLER], rot=0)
    eksen.axhline(0, color=MUREKKEP, linewidth=1)
    eksen.set(title="Kritik kapsam seyrekleşince paket süresi nasıl değişti?",
              xlabel="Başlangıç test sayısı", ylabel="B − A · eşleşen koşumların ortalaması (dk)")
    eksen.legend(title=None, frameon=False)
    eksen.grid(axis="y", color="#E4EAF2")
    eksen.set_axisbelow(True)
    eksen.figure.tight_layout()
    plt.show()
