"""Notebook için veri özeti ve deney planı grafikleri."""

from math import isfinite, log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from kod.veri.data_generator import generate_telecom_data


def deney_alanini_ciz(boyutlar, veri_tohumlari, tekrar, agirliklar,
                     populasyon, iterasyon, komsu_deneme, ga_nesil, *,
                     spiral_deneme=0, havuz_sayisi=1):
    if not all(isfinite(a) and a >= 0 for a in agirliklar) or sum(agirliklar) <= 0:
        raise ValueError("Amaç ağırlıkları negatif olamaz ve toplamları pozitif olmalı.")

    mavi, lacivert, gri, turuncu = "#1169A0", "#1A2A44", "#B5C4D5", "#E57D24"
    satir, sutun = len(veri_tohumlari), len(boyutlar)
    kosum_sayisi = havuz_sayisi * (2 * tekrar + 1)
    toplam_kosum = satir * sutun * kosum_sayisi
    fig, ((izgara, hedef), (butce, secim)) = plt.subplots(
        2, 2, figsize=(max(12.2, 7.3 + 1.25 * sutun), max(7.2, 5.5 + 0.48 * satir)),
        layout="constrained", facecolor="white",
    )
    fig.suptitle(
        f"Deney alanı  ·  {sutun} test boyutu  ×  {satir} veri tohumu  ·  {toplam_kosum} koşum",
        fontsize=16, fontweight="bold", color=lacivert,
    )

    izgara.imshow(np.ones((satir, sutun)), cmap="Blues", vmin=0, vmax=4, aspect="auto")
    izgara.set_xticks(range(sutun), boyutlar)
    izgara.set_yticks(range(satir), veri_tohumlari)
    izgara.set_xticks(np.arange(sutun + 1) - 0.5, minor=True)
    izgara.set_yticks(np.arange(satir + 1) - 0.5, minor=True)
    izgara.grid(which="minor", color="white", linewidth=2)
    izgara.tick_params(which="both", length=0)
    for i in range(satir):
        for j in range(sutun):
            izgara.text(j, i, f"{kosum_sayisi} koşum", ha="center", va="center",
                        color=lacivert, fontsize=10, fontweight="bold")
    izgara.set_title(f"Deney matrisi · her hücrede {havuz_sayisi} havuz türü")
    izgara.set_xlabel("Başlangıç test sayısı")
    izgara.set_ylabel("Veri tohumu")

    paylar = [100 * a / sum(agirliklar) for a in agirliklar]
    olcutler = ["Süre", "Test sayısı", "Kapsam tekrarı", "Nadir hedef kaybı"][:len(paylar)]
    cubuklar = hedef.barh(olcutler, paylar, color=[mavi, "#468BB5", "#87ABC3", "#8250B5"])
    hedef.invert_yaxis()
    hedef.bar_label(cubuklar, labels=[f"%{a:g}" for a in paylar], padding=5)
    hedef.set_xlim(0, 113)
    hedef.set_xticks([0, 25, 50, 75, 100])
    hedef.set_title("Amaç fonksiyonu ağırlıkları")
    hedef.set_xlabel("Normalize edilmiş ağırlık (%)")

    taban = np.zeros(2, dtype=int)
    bolumler = [
        ("Başlangıç", [populasyon, populasyon], gri),
        ("MGO hareketleri", [4 * populasyon * iterasyon, 0], mavi),
        ("Yerel komşuluk", [komsu_deneme * iterasyon, 0], "#3F9787"),
        ("GA nesilleri", [0, populasyon * ga_nesil], turuncu),
    ]
    if spiral_deneme:
        bolumler.insert(3, ("Spiral arama", [spiral_deneme * iterasyon, 0], "#8250B5"))
    for etiket, degerler, renk in bolumler:
        butce.barh(["IBMGO", "GA"], degerler, left=taban, color=renk, label=etiket)
        taban += degerler
    for i, toplam in enumerate(taban):
        butce.text(toplam + max(taban) * 0.025, i, str(toplam), va="center", color=lacivert)
    butce.invert_yaxis()
    butce.set_xlim(0, max(taban) * 1.18)
    butce.set_title(f"Koşum başına planlanan bütçe · {populasyon} birey, {iterasyon} IBMGO turu")
    butce.set_xlabel("Aday değerlendirmesi")
    butce.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                 frameon=False, fontsize=9)

    log_paket = [n * log10(2) for n in boyutlar]
    secim_cubuklari = secim.barh([f"{n} test" for n in boyutlar], log_paket, color=mavi)
    secim.invert_yaxis()
    secim.bar_label(secim_cubuklari, labels=[f"2^{n}" for n in boyutlar], padding=5)
    secim.set_xlim(0, max(log_paket) * 1.18)
    secim.set_title("Teorik test paketi uzayı")
    secim.set_xlabel("Paket sayısının log₁₀ değeri · kapsam ve ön koşul filtresinden önce")

    for eksen in (hedef, butce, secim):
        eksen.set_axisbelow(True)
        eksen.grid(axis="x", color="#DFE6EF", linewidth=0.8)
        for kenar in ("top", "right", "left"):
            eksen.spines[kenar].set_visible(False)
    plt.show()



# Üretilen test havuzunun kapsam, süre ve ön koşul özetini göster.
def deney_verisini_goster(boyutlar, veri_tohumlari, profil):
    ozet_satirlari, buyuk_havuzlar = [], []
    en_buyuk = max(boyutlar)
    for n in boyutlar:
        for tohum in veri_tohumlari:
            kapsam, sureler, _, _, bagimlilik, bilgi = generate_telecom_data(
                n, seed=tohum, scenario_profile=profil, include_metadata=True,
            )
            kritik_sayi = len(bilgi["critical_feature_indices"])
            ozet_satirlari.append({
                "Test": n, "Veri tohumu": tohum, "Kapsam hedefi": kapsam.shape[1],
                "Kritik hedef": kritik_sayi, "Ön koşul bağlantısı": int(bagimlilik.sum()),
                "Tam paket (dk)": round(float(sureler.sum()) / 60, 1),
            })
            if n == en_buyuk:
                buyuk_havuzlar.append((kapsam, sureler, bagimlilik, bilgi))

    display(pd.DataFrame(ozet_satirlari))
    mavi, koyu, acik = "#1169A0", "#1A2A44", "#C7D7E8"
    palet = ("#087F8C", "#8250B5", "#C75A15", "#2876B9", "#A13770")
    renkler = [palet[i % len(palet)] for i in range(len(buyuk_havuzlar))]
    fig, ((sure, hedef), (matris, akis)) = plt.subplots(
        2, 2, figsize=(13.2, max(8, 7.2 + 0.3 * len(veri_tohumlari))),
        layout="constrained", facecolor="white",
    )
    fig.suptitle(f"Test havuzu · {en_buyuk} test · {profil} profil",
                 fontsize=16, fontweight="bold", color=koyu)
    etiketler = [str(t) for t in veri_tohumlari]
    sureler = [h[1] for h in buyuk_havuzlar]
    kutular = sure.boxplot(
        sureler, vert=False, patch_artist=True,
        medianprops={"color": koyu, "linewidth": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    for kutu, renk in zip(kutular["boxes"], renkler):
        kutu.set(facecolor=renk, edgecolor=renk, alpha=.55)
    toplamlar = [float(d.sum()) / 60 for d in sureler]
    sure.set_yticks(range(1, len(etiketler) + 1), [
        f"{t} · {toplam:.1f} dk" for t, toplam in zip(etiketler, toplamlar)
    ])
    sure.invert_yaxis()
    sure.set_title("Test süreleri ve tam paket toplamı")
    sure.set_xlabel("Bir testin tahmini süresi (sn)")
    sure.set_ylabel("Tohum · toplam süre")
    sure.set_axisbelow(True)
    sure.grid(axis="x", color="#DFE6EF")

    kritik = np.array([len(h[3]["critical_feature_indices"]) for h in buyuk_havuzlar])
    tum = np.array([h[0].shape[1] for h in buyuk_havuzlar])
    hedef.barh(etiketler, kritik, color=renkler, label="Kritik")
    hedef.barh(etiketler, tum - kritik, left=kritik, color=renkler, alpha=.25, label="Diğer")
    hedef.invert_yaxis()
    for i, (k, t) in enumerate(zip(kritik, tum)):
        hedef.text(k / 2, i, str(k), ha="center", va="center", color="white", fontweight="bold")
        fark = f" ({t - tum[0]:+d})" if i else " (baz)"
        hedef.text(t + max(tum) * 0.018, i, f"{t}{fark}", va="center", color=koyu)
    hedef.set_xlim(0, max(tum) * 1.26)
    hedef.set_title(f"Kapsam hedefleri · baz tohum {etiketler[0]}")
    hedef.set_xlabel("Hedef sayısı")
    hedef.set_ylabel("Veri tohumu")
    hedef.legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, frameon=False, fontsize=9)

    kapsam, _, bagimlilik, bilgi = buyuk_havuzlar[0]
    # Veri bölümündeki kampanya örneğinin satır ve sütunları.
    test_etiketleri = {
        "T_0001": "İlk aktivasyon",
        "T_0004": "Tekrar başvuru",
        "T_0005": "İlk fatura",
        "T_0006": "Süre bitimi",
    }
    kural_etiketleri = {
        "G05": "Tek hak",
        "G06": "Tekrar reddi",
        "G07": "İlk fatura",
        "G08": "Yarar sonu",
        "KMP01_A": "Tekil aktivasyon",
    }
    test_indeksleri = {v["id"]: i for i, v in enumerate(bilgi["test_cases"])}
    kural_indeksleri = {v["id"]: i for i, v in enumerate(bilgi["requirements"])}
    satirlar = [test_indeksleri[k] for k in test_etiketleri]
    sutunlar = [kural_indeksleri[k] for k in kural_etiketleri]
    ornek = kapsam[np.ix_(satirlar, sutunlar)]
    matris.imshow(ornek, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    matris.set_yticks(range(len(satirlar)), [f"{k} · {v}" for k, v in test_etiketleri.items()])
    matris.set_xticks(range(len(sutunlar)), [f"{k}\n{v}" for k, v in kural_etiketleri.items()],
                      fontsize=9)
    for (i, j), deger in np.ndenumerate(ornek):
        matris.text(j, i, str(deger), color="white" if deger else koyu,
                    ha="center", va="center", fontweight="bold")
    matris.set_title(f"Örnek kapsam matrisi · veri tohumu {veri_tohumlari[0]}")
    matris.set_xlabel("1 = test ilgili kuralı kapsar")

    akis.set(xlim=(0, 1), ylim=(0, 1), title="Örnek ön koşul akışı")
    akis.axis("off")
    konumlar = [(0.22, .52), (.78, .80), (.78, .52), (.78, .24)]
    for i, kaynak in enumerate(satirlar):
        for j, hedef_test in enumerate(satirlar):
            if bagimlilik[kaynak, hedef_test]:
                x0, y0 = konumlar[i]
                x1, y1 = konumlar[j]
                akis.annotate("", xy=(x1 - .15, y1), xytext=(x0 + .15, y0),
                              arrowprops={"arrowstyle": "->", "color": mavi, "lw": 1.8})
    for (x, y), (kimlik, ad), i in zip(konumlar, test_etiketleri.items(), satirlar):
        akis.text(x, y, f"{ad}\n{kimlik}", ha="center", va="center",
                  color=koyu, fontsize=10,
                  bbox={"boxstyle": "round,pad=0.55", "facecolor": "#EAF3F9", "edgecolor": acik})
    akis.text(.5, .03, "Ok, önce çalışması gereken testten bağımlı teste gider.",
              ha="center", color=koyu, fontsize=9)
    plt.show()

    if len(buyuk_havuzlar) > 1:
        # Aynı test kimliklerini ilk veri tohumuyla karşılaştır.
        referans = bilgi["test_cases"]
        alanlar = {
            "Tahmini süre": ("duration_sec",),
            "Başvuru kanalı": ("channel",),
            "Senaryo": ("offer_code", "family", "expected_result", "requirement_ids"),
            "Ön koşullar": ("prerequisite_test_ids",),
        }
        fig, eksen = plt.subplots(figsize=(10, 3.8), layout="constrained")
        konumlar = np.arange(len(alanlar))
        genislik = 0.8 / (len(buyuk_havuzlar) - 1)
        for i, havuz in enumerate(buyuk_havuzlar[1:]):
            testler = {test["id"]: test for test in havuz[3]["test_cases"]}
            oranlar = [
                100 * sum(
                    any(test[alan] != testler[test["id"]][alan] for alan in alan_grubu)
                    for test in referans
                ) / en_buyuk
                for alan_grubu in alanlar.values()
            ]
            cubuklar = eksen.barh(
                konumlar - 0.4 + (i + 0.5) * genislik, oranlar, genislik,
                label=f"{veri_tohumlari[i + 1]} ↔ {veri_tohumlari[0]}", color=renkler[i + 1],
            )
            eksen.bar_label(cubuklar, labels=[f"%{oran:.1f}" for oran in oranlar], padding=3)
        eksen.set(
            yticks=konumlar, yticklabels=list(alanlar), xlim=(0, 112),
            xlabel="Değişen testlerin oranı (%)",
            title=f"Veri tohumu değişince ne değişiyor? · {en_buyuk} test",
        )
        eksen.set_xticks(range(0, 101, 20))
        eksen.invert_yaxis()
        eksen.legend(title="Veri tohumları", loc="lower right")
        plt.show()
