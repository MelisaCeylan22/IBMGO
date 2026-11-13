"""Nadir hedef ölçütünü iki küçük test paketiyle göster."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def nadir_hedef_ornegini_goster(puanla):
    ornek_kapsam = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
    ornek_sureler = [10, 10, 10]
    paketler = {"Nadir hedefi kapsayan": [1, 0, 0], "Yaygın hedefi kapsayan": [0, 1, 0]}
    puanlar = []
    for ad, paket in paketler.items():
        girdiler = (paket, ornek_kapsam, ornek_sureler)
        kisitlar = dict(critical_feature_indices=[0], min_coverage=2 / 3)
        eski = puanla(*girdiler, **kisitlar)
        yeni = puanla(
            *girdiler, **kisitlar, w_time=0.40, w_count=0.30,
            w_redundancy=0.15, w_rare=0.15,
        )
        puanlar.append({"Paket": ad, "Standart puan": eski, "Nadirlik eklenince": yeni})
    puan_tablosu = pd.DataFrame(puanlar)
    display(puan_tablosu.round(3))

    eksen = puan_tablosu.plot.barh(
        x="Paket", color=["#64748B", "#087F8C"], figsize=(9, 3.3),
        title="Nadir hedefi korumak puanı nasıl değiştirir?",
    )
    for cubuklar in eksen.containers:
        eksen.bar_label(cubuklar, fmt="%.3f", padding=4)
    eksen.set(xlabel="Amaç puanı · düşük değer daha iyi", ylabel="")
    eksen.margins(x=0.25)
    eksen.invert_yaxis()
    eksen.legend(loc="upper right")
    eksen.figure.tight_layout()
    plt.show()
