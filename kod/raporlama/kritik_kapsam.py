"""Kritik kapsam duyarlılığı için özet, hedef tablosu ve grafik."""

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from kod.veri.data_generator import generate_telecom_data
from kod.veri.kapsam_duyarliligi import kritik_kapsami_seyrelt


def kritik_kapsam_karsilastirmasini_goster(n, baz_tohum, profil, *, adim=4, hedef_sayisi=8):
    kapsam_a, sureler, _, _, on_kosullar, bilgi = generate_telecom_data(
        n, seed=baz_tohum, scenario_profile=profil, include_metadata=True,
    )
    kritik = bilgi["critical_feature_indices"]
    kapsam_b = kritik_kapsami_seyrelt(kapsam_a, kritik, adim)
    once = kapsam_a[:, kritik].sum(axis=0)
    sonra = kapsam_b[:, kritik].sum(axis=0)
    assert ((once > 0) == (sonra > 0)).all()

    ozet = pd.DataFrame({
        "Havuz": ["A · Orijinal", "B · Seyrek kritik kapsam"],
        "Test": [n, n],
        "Toplam hedef": [kapsam_a.shape[1], kapsam_b.shape[1]],
        "Kapsanan kritik hedef": [(once > 0).sum(), (sonra > 0).sum()],
        "Ön koşul bağlantısı": [int(on_kosullar.sum())] * 2,
        "Toplam süre (dk)": [round(sureler.sum() / 60, 1)] * 2,
        "Kritik kapsam eşleşmesi": [once.sum(), sonra.sum()],
    })
    display(ozet)

    hedefler = pd.DataFrame({
        "Hedef": [bilgi["requirements"][i]["id"] for i in kritik],
        "A · Orijinal": once,
        "B · Seyrek": sonra,
    })
    hedefler["Azalma"] = once - sonra
    ornek = hedefler.sort_values("Azalma", ascending=False).head(hedef_sayisi)
    display(ornek)
    fig, (ornek_ekseni, dagilim_ekseni) = plt.subplots(1, 2, figsize=(14, 4.8))
    eksen = ornek.plot.barh(
        x="Hedef", y=["A · Orijinal", "B · Seyrek"],
        color=["#087F8C", "#C75A15"], ax=ornek_ekseni,
        title="Kapsayan test sayısı en çok azalan hedefler",
    )
    eksen.set_xlabel("Kapsayan test sayısı")
    eksen.invert_yaxis()
    for cubuklar in eksen.containers:
        eksen.bar_label(cubuklar, fmt="%d", padding=4)
    eksen.margins(x=0.14)
    fig.suptitle(f"Kritik kapsamın seyrekleşmesi  ·  {n:,} test  ·  Tohum {baz_tohum}",
                 fontsize=16, fontweight="bold")
    fig.text(
        0.5, 0.03,
        f"En çok azalan {len(ornek)} hedef gösteriliyor. "
        f"İki havuz da aynı {(once > 0).sum()} kritik hedefi kapsıyor.",
        ha="center", fontsize=10,
    )

    # Her kritik hedefi, onu kapsayan test sayısına göre bir gruba yerleştir.
    sinirlar = [-1, 0, 1, 5, 20, 100, max(n, 101)]
    gruplar = ["0", "1", "2–5", "6–20", "21–100", "101+"]
    dagilim = pd.DataFrame({
        ad: pd.cut(sayilar, sinirlar, labels=gruplar).value_counts()
        for ad, sayilar in [("A · Orijinal", once), ("B · Seyrek", sonra)]
    })
    eksen = dagilim.plot.bar(
        color=["#087F8C", "#C75A15"], ax=dagilim_ekseni, rot=0,
        title="Tüm kritik hedeflerin dağılımı",
    )
    eksen.set(xlabel="Bir hedefi kapsayan test sayısı", ylabel="Kritik hedef sayısı")
    for cubuklar in eksen.containers:
        eksen.bar_label(cubuklar, labels=[str(int(v)) if v else "" for v in cubuklar.datavalues],
                       padding=3)
    eksen.margins(y=0.15)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    plt.show()
