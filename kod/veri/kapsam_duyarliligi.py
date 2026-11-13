"""Kritik kapsamın seyrekleştiği sentetik deney havuzu."""

import numpy as np


def kritik_kapsami_seyrelt(kapsam, kritik_indeksler, adim=4):
    """Kritik hedef başına kapsayan testlerin her adımda birini tut."""
    if not isinstance(adim, int) or adim < 1:
        raise ValueError("Seyreltme adımı pozitif tam sayı olmalı.")
    seyrek = kapsam.copy()
    for hedef in kritik_indeksler:
        testler = np.flatnonzero(kapsam[:, hedef])
        # İlk eşleşmeyi koru, böylece kapsanan hiçbir hedef kaybolmaz.
        seyrek[testler[np.arange(len(testler)) % adim != 0], hedef] = 0
    return seyrek
