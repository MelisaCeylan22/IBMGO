"""Notebook deney planı ve yöntem karşılaştırması."""

from math import isfinite
from time import perf_counter

import pandas as pd

from kod.algoritmalar.baselines import genetic_algorithm_optimizer, greedy_regression_selector
from kod.algoritmalar.bmgo import binary_mountain_gazelle_optimizer, dimension_adaptive_settings
from kod.algoritmalar.fitness import get_suite_metrics
from kod.veri.data_generator import generate_telecom_data
from kod.veri.kapsam_duyarliligi import kritik_kapsami_seyrelt


YONTEMLER = ("IBMGO", "GA", "Greedy")


def deney_planini_hazirla(veri, amac, arama, gosterim):
    """Geçersiz ayarları veri üretimi ve arama başlamadan bildir."""
    boyutlar, tohumlar = veri["boyutlar"], veri["tohumlar"]
    if not boyutlar or any(type(n) is not int or n < 50 for n in boyutlar) or len(set(boyutlar)) != len(boyutlar):
        raise ValueError("Test boyutları en az 50 olan farklı tam sayılar olmalı.")
    if not tohumlar or any(type(t) is not int or t < 0 for t in tohumlar) or len(set(tohumlar)) != len(tohumlar):
        raise ValueError("Veri tohumları negatif olmayan farklı tam sayılar olmalı.")
    if veri["profil"] not in {"standard", "expanded"}:
        raise ValueError("Profil 'standard' veya 'expanded' olmalı.")
    if type(veri["seyreltme_adimi"]) is not int or veri["seyreltme_adimi"] < 1:
        raise ValueError("Seyreltme adımı pozitif tam sayı olmalı.")

    havuzlar = arama["havuzlar"]
    tohumlar = arama["algoritma_tohumlari"]
    if not havuzlar or not set(havuzlar) <= {"A", "B"} or len(set(havuzlar)) != len(havuzlar):
        raise ValueError("Deney havuzları ['A'], ['B'] veya ['A', 'B'] olmalı.")
    if not tohumlar or any(type(t) is not int or t < 0 for t in tohumlar) or len(set(tohumlar)) != len(tohumlar):
        raise ValueError("Algoritma tohumları negatif olmayan farklı tam sayılar olmalı.")
    if not 0 < amac["asgari_kapsam"] <= 1:
        raise ValueError("Asgari genel kapsam 0'dan büyük, 1'e eşit veya küçük olmalı.")
    agirliklar = list(amac["agirliklar"].values())
    if any(not isfinite(a) or a < 0 for a in agirliklar) or sum(agirliklar) <= 0:
        raise ValueError("Ağırlıklar negatif olamaz ve toplamları pozitif olmalı.")
    for anahtar in ("kapsam_hedef_sayisi", "oncelik_test_sayisi", "havuz_sayfa_boyu"):
        if type(gosterim[anahtar]) is not int or gosterim[anahtar] < 1:
            raise ValueError(f"{anahtar} pozitif tam sayı olmalı.")
    if not isfinite(gosterim["kritik_oncelik_agirligi"]) or gosterim["kritik_oncelik_agirligi"] < 1:
        raise ValueError("Kritik öncelik ağırlığı en az 1 olmalı.")
    baz_tohum = gosterim["kapsam_tohumu"]
    if baz_tohum is not None and baz_tohum not in veri["tohumlar"]:
        raise ValueError("2B örnek tohumu veri tohumları arasında olmalı.")
    ornek_boyut = gosterim["kapsam_boyutu"]
    if ornek_boyut is not None and ornek_boyut not in boyutlar:
        raise ValueError("2B örnek boyutu test boyutları arasında olmalı.")

    populasyon = arama["populasyon"]
    tur = arama["ibmgo_tur"]
    if type(populasyon) is not int or populasyon < 2 or type(tur) is not int or tur < 1:
        raise ValueError("Popülasyon en az 2, IBMGO tur sayısı en az 1 olmalı.")
    ibmgo = {k: v for k, v in arama["ibmgo"].items() if v is not None}
    ibmgo.setdefault("local_search_trials", populasyon)
    ibmgo.setdefault("spiral_search_trials", 0)
    yerel, spiral = ibmgo["local_search_trials"], ibmgo["spiral_search_trials"]
    if any(type(v) is not int or v < 0 for v in (yerel, spiral)):
        raise ValueError("Yerel ve spiral deneme sayıları negatif olmayan tam sayılar olmalı.")
    ga = arama["ga"].copy()
    nesil = ga.pop("max_generations")
    ibmgo_butcesi = populasyon + tur * (4 * populasyon + yerel + spiral)
    if nesil is None:
        nesil = max(1, round((ibmgo_butcesi - populasyon) / populasyon))
    if type(nesil) is not int or nesil < 1:
        raise ValueError("GA nesil sayısı pozitif tam sayı veya otomatik olmalı.")
    return dict(ibmgo=ibmgo, ga=ga, ga_nesil=nesil,
                ibmgo_butcesi=ibmgo_butcesi, ga_butcesi=populasyon * (1 + nesil))


def deneyleri_calistir(veri, amac, arama, plan):
    """Aynı havuz ve tohumlar üzerinde üç yöntemi çalıştır."""
    algoritma_tohumlari = arama["algoritma_tohumlari"]
    populasyon = arama["populasyon"]
    asgari_kapsam = amac["asgari_kapsam"]
    kosumlar = []
    ornek_secim = None

    for test_sayisi in veri["boyutlar"]:
        for veri_tohumu in veri["tohumlar"]:
            kapsam, sureler, adlar, _, bagimliliklar, bilgi = generate_telecom_data(
                test_sayisi, seed=veri_tohumu, scenario_profile=veri["profil"],
                include_metadata=True,
            )
            kritik = bilgi["critical_feature_indices"]
            ortak = dict(
                coverage_matrix=kapsam, durations=sureler, dependency_matrix=bagimliliklar,
                critical_feature_indices=kritik, min_coverage=asgari_kapsam,
                **amac["agirliklar"],
            )
            ibmgo_ayarlari = dimension_adaptive_settings(test_sayisi) if arama["boyuta_adapte"] else {}
            ibmgo_ayarlari.update(plan["ibmgo"])

            for havuz in arama["havuzlar"]:
                havuz_kapsami = (
                    kritik_kapsami_seyrelt(kapsam, kritik, veri["seyreltme_adimi"])
                    if havuz == "B" else kapsam
                )
                havuz_girdileri = dict(ortak, coverage_matrix=havuz_kapsami)
                for yontem in YONTEMLER:
                    tohumlar = [None] if yontem == "Greedy" else algoritma_tohumlari
                    for tohum in tohumlar:
                        baslama = perf_counter()
                        if yontem == "IBMGO":
                            secim, puan, _ = binary_mountain_gazelle_optimizer(
                                **havuz_girdileri, **ibmgo_ayarlari, pop_size=populasyon,
                                max_iter=arama["ibmgo_tur"], random_seed=tohum,
                            )
                        elif yontem == "GA":
                            secim, puan, _ = genetic_algorithm_optimizer(
                                **havuz_girdileri, **plan["ga"], pop_size=populasyon, max_generations=plan["ga_nesil"],
                                random_seed=tohum,
                            )
                        else:
                            secim, puan = greedy_regression_selector(**havuz_girdileri)
                        gecen = perf_counter() - baslama

                        metrik = get_suite_metrics(secim, havuz_kapsami, sureler, bagimliliklar, kritik)
                        if (metrik["critical_missing"] or metrik["broken_dependencies"]
                                or metrik["coverage_ratio"] < asgari_kapsam - 1e-9):
                            raise RuntimeError(f"Geçersiz paket: Havuz {havuz}, {yontem}, {test_sayisi} test, veri tohumu {veri_tohumu}.")

                        kosumlar.append({
                            "Havuz": havuz, "Test sayısı": test_sayisi, "Veri tohumu": veri_tohumu,
                            "Algoritma tohumu": tohum, "Yöntem": yontem, "Fitness": puan,
                            "Tüm paket (dk)": metrik["total_duration"] / 60,
                            "Seçili paket (dk)": metrik["selected_duration"] / 60,
                            "Tasarruf (%)": 100 * (1 - metrik["time_ratio"]),
                            "Seçilen test": metrik["selected_count"],
                            "Kritik kapsam (%)": 100 * (1 - metrik["critical_missing"] / len(kritik)),
                            "Genel kapsam (%)": 100 * metrik["coverage_ratio"],
                            "Ön koşul ihlali": metrik["broken_dependencies"],
                            "Paket seçme (sn)": gecen,
                        })
                        if yontem == "IBMGO" and ornek_secim is None:
                            ornek_secim = (havuz, secim, havuz_kapsami, sureler, adlar, bagimliliklar,
                                           kritik, test_sayisi, veri_tohumu, tohum)

                print(f"Havuz {havuz} · {test_sayisi} test · veri tohumu {veri_tohumu} tamamlandı.")

    return pd.DataFrame(kosumlar), ornek_secim
