"""Tam test havuzu için etkileşimli grafikler ve test kayıtları."""

from io import BytesIO

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from kod.veri.data_generator import generate_telecom_data


def _test_havuzu_paneli(n, veri_tohumu, profil, sayfa_boyu):
    kapsam, sureler, adlar, _, bagimlilik, bilgi = generate_telecom_data(
        n, seed=veri_tohumu, scenario_profile=profil, include_metadata=True,
    )
    hedefler = [r["id"] for r in bilgi["requirements"]]
    # Her dolu hücrenin satırı ön koşulu, sütunu ona bağlı testi gösterir.
    once, sonra = np.nonzero(bagimlilik)
    renk = "#1169A0"
    fig = plt.figure(figsize=(12.6, 7.2),
                     layout="constrained", facecolor="white")
    bolumler = fig.add_gridspec(2, 2, width_ratios=[1, 1.12])
    sure_ekseni = fig.add_subplot(bolumler[0, 0])
    on_kosul_ekseni = fig.add_subplot(bolumler[1, 0])
    kapsam_ekseni = fig.add_subplot(bolumler[:, 1])
    fig.suptitle(f"{n} test · Veri tohumu {veri_tohumu}", fontsize=17, weight="bold", color="#1A2A44")

    sure_ekseni.scatter(np.arange(1, n + 1), sureler, s=8, color=renk, alpha=.65)
    sure_ekseni.set(title="Tüm testlerin tahmini süreleri", xlabel="Test sırası",
                    ylabel="Süre (sn)", xlim=(.5, n+.5))
    sure_ekseni.set_axisbelow(True)
    sure_ekseni.grid(axis="y", color="#E2E8F0")

    on_kosul_ekseni.scatter(once + 1, sonra + 1, s=12, color="#D56A20", alpha=.8)
    on_kosul_ekseni.set(title=f"Ön koşul bağlantıları · {len(once)}",
                       xlabel="Önce yer alan test", ylabel="Bağımlı test",
                       xlim=(.5, n+.5), ylim=(n+.5, .5))
    on_kosul_ekseni.set_axisbelow(True)
    on_kosul_ekseni.grid(color="#E2E8F0")

    kapsam_ekseni.imshow(
        kapsam, cmap=ListedColormap(["#F3F6FA", renk]), vmin=0, vmax=1,
        interpolation="nearest", aspect="auto",
        extent=(.5, len(hedefler)+.5, n+.5, .5),
    )
    kapsam_ekseni.set(title=f"Kapsam matrisi · {n} test × {len(hedefler)} hedef",
                     xlabel="Kapsam hedefi sırası · mavi = kapsıyor", ylabel="Test sırası")

    grafik = BytesIO()
    fig.savefig(grafik, format="png", dpi=165, facecolor="white")
    plt.close(fig)

    kayitlar = pd.DataFrame([
        {
            "Test": vaka["id"],
            "Senaryo": adlar[i],
            "Süre (sn)": int(sureler[i]),
            "Ön koşul": ", ".join(vaka["prerequisite_test_ids"]) or "—",
            "Kapsam hedefleri": ", ".join(hedefler[k] for k in np.flatnonzero(kapsam[i])),
            "Beklenen sonuç": vaka["expected_result"],
        }
        for i, vaka in enumerate(bilgi["test_cases"])
    ])
    aralik = widgets.Dropdown(
        options=[(f"{bas+1}–{min(bas+sayfa_boyu, n)}", bas) for bas in range(0, n, sayfa_boyu)],
        description="Test aralığı", style={"description_width": "initial"},
    )
    tablo = widgets.HTML()

    def araligi_goster(change=None):
        bas = aralik.value
        secili = kayitlar.iloc[bas:bas + sayfa_boyu]
        tablo.value = (
            f"<p><strong>{bas+1}–{bas+len(secili)} / {n} test kaydı</strong></p>"
            '<div style="overflow:auto;max-height:420px">'
            + secili.to_html(index=False, border=0, escape=True)
            + "</div>"
        )

    aralik.observe(araligi_goster, names="value")
    araligi_goster()
    bilgi_satiri = widgets.HTML(
        f"<p><strong>{n} test</strong> · {len(hedefler)} kapsam hedefi · "
        f"{len(bilgi['critical_feature_indices'])} kritik hedef · "
        f"{float(sureler.sum())/60:.1f} dk tahmini tam paket süresi. "
        f"Teorik seçim uzayı 2<sup>{n}</sup> paket.</p>"
    )
    return widgets.VBox([
        bilgi_satiri,
        widgets.Image(value=grafik.getvalue(), format="png",
                      layout=widgets.Layout(width="100%", max_width="1250px")),
        aralik, tablo,
    ])


def tam_test_havuzu(boyutlar, veri_tohumlari, profil, *, sayfa_boyu=20):
    """Yalnızca seçili havuzu üretip göster."""
    secim = widgets.Dropdown(
        options=[(f"{n} test · Tohum {t}", (n, t))
                 for n in boyutlar for t in veri_tohumlari],
        description="Havuz",
    )
    gorunum = widgets.VBox()

    def havuzu_guncelle(change=None):
        n, tohum = secim.value
        gorunum.children = (_test_havuzu_paneli(n, tohum, profil, sayfa_boyu),)

    secim.observe(havuzu_guncelle, names="value")
    havuzu_guncelle()
    return widgets.VBox([secim, gorunum])
