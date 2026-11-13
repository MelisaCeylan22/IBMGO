"""Telekom kampanya / taahhüt test paketleri için kural tabanlı sentetik veri.

Bu veri üreticisi gerçek bir operatörün sistemine veya müşteri verilerine bağlı
değildir. Kampanya davranışları bir ders projesi için kurgulanmıştır.

BMGO'ya uyumlu temel dönüş:
    coverage_matrix, durations, test_names, features, dependency_matrix

İsteğe bağlı metadata ile senaryonun girdisi, beklenen sonucu ve kritiklik
bilgisi de alınabilir. ``main.py`` içindeki mevcut fonksiyona doğrudan
bağlamak için ``generate_telecom_data`` kullanılabilir.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Offer:
    code: str
    name: str
    months: int
    monthly_discount: float
    data_gb: int
    package_fee: int
    customer_group: str
    minimum_tenure: int
    commitment_required: bool = False


OFFERS = (
    Offer("KMP01", "12 Ay %20 Indirim", 12, 0.20, 0, 0, "faturali", 0),
    Offer("KMP02", "Hos Geldin 5 GB", 1, 0.00, 5, 0, "yeni", 0),
    Offer("KMP03", "Ek 10 GB Paketi", 1, 0.00, 10, 50, "herkes", 0),
    Offer("KMP04", "Sadakat %15 Indirim", 6, 0.15, 0, 0, "sadakat", 6),
    Offer("KMP05", "Taahhut Yenileme", 12, 0.10, 0, 0, "faturali", 6, True),
)

# Genişletilmiş profilde ilk 50 örnekten sonra kullanılan ek teklif aileleri.
# İlk 50 senaryonun eski dosyalarla birebir uyumunu korumak için bu liste
# ayrı tutulur; varyantlarda beş test-aile bloğuna dönüşümlü dağıtılır.
EXTRA_OFFERS = (
    Offer("KMP06", "Numara Tasima 15 GB", 3, 0.00, 15, 0, "yeni", 0),
    Offer("KMP07", "Aile Ikinci Hat %10", 12, 0.10, 0, 0, "aile", 0),
    Offer("KMP08", "Genc 20 GB", 1, 0.00, 20, 80, "genc", 0),
    Offer("KMP09", "Yurt Disi 3 GB", 1, 0.00, 3, 90, "herkes", 0),
)

# Her teklif için on senaryo vardır. Sadakat ve taahhüt yenileme tekliflerindeki
# birer örnek, sektöre özgü hedef kitle ve taahhüt uygunluğu kontrolleridir.
# 5 teklif x 10 davranış = 50 temel senaryo.
FAMILIES = (
    "activation_success",
    "inactive_rejected",
    "window_closed",
    "repeat_rejected",
    "first_bill",
    "expired_benefit",
    "early_cancel",
    "idempotent_request",
    "last_valid_bill",
    "invalid_offer_code",
)


def _families_for_offer(offer: Offer) -> tuple[str, ...]:
    if offer.code == "KMP04":
        return (*FAMILIES[:-1], "not_targeted")
    if offer.code == "KMP05":
        return (*FAMILIES[:-1], "commitment_ineligible")
    return FAMILIES


def _families_for_variant(
    offer: Offer, variant: int, expanded: bool = False
) -> tuple[str, ...]:
    """Yeni varyantlarda yeni hata ve sınır koşulları ekler.

    On temel akışın hepsi ilk 50 vakada temsil edilir; sonraki gruplarda
    tümüyle aynı akışlar yerine sektörel sınır senaryoları da bulunur.
    """
    cases = list(_families_for_offer(offer))
    if variant >= 1:
        cases[9] = "boundary_last_day"
    if variant >= 2 and variant % 2 == 0:
        cases[7] = "concurrent_activation"
    if variant >= 3 and variant % 3 == 0:
        cases[5] = "payment_failure" if offer.package_fee > 0 else (
            "quota_exhaustion" if offer.data_gb > 0 else "conflicting_discount"
        )
    if variant >= 4 and variant % 4 == 0:
        if offer.data_gb > 0:
            cases[6] = "quota_exhaustion"
        elif offer.code == "KMP04":
            cases[6] = "tenure_rejected"
    if expanded and variant >= 5:
        # Aynı kampanya, yeni müşteri koşullarında farklı hata yollarına sahip.
        cases[2] = ("auth_expired", "portability_pending", "blacklisted")[
            variant % 3
        ]
        if variant % 4 == 0:
            cases[4] = "duplicate_billing"
        elif variant % 4 == 1 and offer.monthly_discount > 0:
            cases[4] = "prorated_upgrade"
        elif variant % 4 == 2 and offer.data_gb > 0:
            cases[4] = "quota_reset"
        if variant % 5 == 0:
            cases[8] = "ownership_changed"
    if expanded and variant >= 2:
        if offer.code == "KMP07" and variant % 3 == 0:
            cases[6] = "second_line_ineligible"
        elif offer.code == "KMP08" and variant % 3 == 1:
            cases[6] = "age_ineligible"
        elif offer.code == "KMP09" and variant % 3 == 2:
            cases[6] = "roaming_disabled"
    return tuple(cases)

COMMON_RULES = (
    ("G01", "Aktif ve uygun musterinin isleme devam edebilmesi", False),
    ("G02", "Pasif hat icin islem reddi", True),
    ("G03", "Basvuru suresi acik teklifin kabul edilmesi", False),
    ("G04", "Basvuru suresi bitmis teklifin reddi", True),
    ("G05", "Ilk aktivasyonda tek hak olusturulmasi", True),
    ("G06", "Ayni donemde tekrar kullanimin reddi", True),
    ("G07", "Ilk fatura doneminde dogru yarar uygulanmasi", True),
    ("G08", "Yarar suresinden sonra yararin sonlanmasi", True),
    ("G09", "Iptal sonrasi yararin uygulanmamasi", True),
    ("G10", "Ayni istek kimliginin ikinci kayit olusturmamasi", True),
    ("G11", "Son gecerli donemde yararin devam etmesi", True),
    ("G12", "Gecersiz teklif kodunun reddi", False),
    ("G13", "Uygun teklif aktivasyonunun basarili olmasi", False),
    ("G14", "Red edilen islemin yarar yaratmamasi", False),
    ("G15", "Hedef musterisi olmayan kisinin kampanyaya alinmamasi", True),
    ("G16", "Taahhut yenilemede uygun sozlesme doneminin aranmasi", True),
)

OFFER_RULES = (
    ("E", "Kampanya uygunluk kurallari (olumlu ve olumsuz)", False),
    ("A", "Kampanya ya da paketin tekil aktivasyonu", True),
    ("B", "Ilk donemde dogru ucret veya kota", True),
    ("L", "Son gecerli donemde dogru yarar", False),
    ("X", "Sure sonunda yararin sonlanmasi", True),
    ("C", "Iptal sonrasi yararin sonlanmasi", True),
)

# İlk 50 testin ötesinde ortaya çıkan, telekoma özgü ilave sınır durumları.
# Bunlar ancak ilgili senaryo veri kümesinde varsa kapsama hedefi olur.
EXTENDED_RULES = (
    ("G17", "Kampanyanin son basvuru gununde aktivasyon kabul edilir", True),
    ("G18", "Eszamanli aktivasyon isteklerinde tek hak olusur", True),
    ("G19", "Odeme basarisizsa ucretli ek paket tanimlanmaz", True),
    ("G20", "Internet kotasi tukendiginde ek kullanim engellenir", True),
    ("G21", "Birbiriyle catisan indirimler ust uste uygulanmaz", True),
    ("G22", "Sadakat kampanyasinda musteri kidem siniri denetlenir", True),
    ("G23", "Suresi dolmus oturumla aktivasyon reddedilir", True),
    ("G24", "Numara tasima tamamlanmadan kampanya acilmaz", True),
    ("G25", "Ara donem tarife degisikliginde ucret dogru hesaplanir", True),
    ("G26", "Ayni fatura olayinda ikinci kez ucret tahakkuk etmez", True),
    ("G27", "Yeni fatura doneminde internet kotasi dogru yenilenir", True),
    ("G28", "Kisitli musteri profiline kampanya acilmaz", True),
    ("G29", "Hat sahipligi degistikten sonra eski kampanya hakki surmez", True),
    ("G30", "Aile paketinde ikinci hat ana hatta bagli olmalidir", True),
    ("G31", "Genc tarifesinde yas uygunlugu denetlenir", True),
    ("G32", "Yurt disi paketinde roaming yetkisi denetlenir", True),
)


def requirement_catalog(extra_offers=()) -> list[dict[str, Any]]:
    """16 ortak + her gözlenen teklif için 6 teklif kuralı."""
    result = [
        {"id": code, "description": desc, "critical": important}
        for code, desc, important in COMMON_RULES
    ]
    for offer in (*OFFERS, *extra_offers):
        for suffix, desc, important in OFFER_RULES:
            result.append(
                {
                    "id": f"{offer.code}_{suffix}",
                    "description": f"{offer.name}: {desc}",
                    "critical": important,
                }
            )
    return result


def _context_catalog(cases: list[TestCase], expanded: bool = False):
    """Kampanya x kanal / müşteri segmenti x iş akışı kapsamını üretir.

    Tek bir testte görülen bağlamlar gereksinim sayılmaz. Bir kapsama
    hedefinin tutulması için en az üç örneğe ve iki ayrı müşteriye dayanması
    gerekir; böylece 1.000 testteki hedefler yalnızca 50 kuralın tekrarı olmaz.
    """
    stage_map = {
        "activation_success": "aktivasyon",
        "boundary_last_day": "aktivasyon",
        "inactive_rejected": "uygunluk",
        "window_closed": "uygunluk",
        "invalid_offer_code": "uygunluk",
        "not_targeted": "uygunluk",
        "commitment_ineligible": "uygunluk",
        "tenure_rejected": "uygunluk",
        "conflicting_discount": "uygunluk",
        "payment_failure": "uygunluk",
        "auth_expired": "uygunluk",
        "portability_pending": "uygunluk",
        "blacklisted": "uygunluk",
        "second_line_ineligible": "uygunluk",
        "age_ineligible": "uygunluk",
        "roaming_disabled": "uygunluk",
        "first_bill": "fatura",
        "prorated_upgrade": "fatura",
        "duplicate_billing": "fatura",
        "quota_reset": "fatura",
        "last_valid_bill": "fatura",
        "expired_benefit": "sure",
        "early_cancel": "sure",
        "quota_exhaustion": "sure",
        "ownership_changed": "sure",
        "repeat_rejected": "tekrar",
        "idempotent_request": "tekrar",
        "concurrent_activation": "tekrar",
    }
    counter: Counter[str] = Counter()
    customers: dict[str, set[str]] = defaultdict(set)
    descriptions: dict[str, str] = {}
    important: dict[str, bool] = {}
    test_tags: list[tuple[str, ...]] = []

    for case in cases:
        stage = stage_map[case.family]
        tenure = case.customer_tenure_months
        segment = "0_1" if tenure <= 1 else "2_5" if tenure <= 5 else "6_11" if tenure <= 11 else "12_plus"
        channel_id = f"CTX_CH_{case.offer_code}_{case.channel}_{stage}"
        segment_id = f"CTX_SEG_{case.offer_code}_{segment}_{stage}"
        tags = [channel_id, segment_id]

        descriptions[channel_id] = f"{case.offer_name}: {case.channel} kanalinda {stage} akisi"
        descriptions[segment_id] = f"{case.offer_name}: {segment} ay musteri kideminde {stage} akisi"
        important[channel_id] = stage == "fatura"
        important[segment_id] = stage == "fatura"
        if expanded:
            # İş kuralı + müşteri/tarife bağlamı birlikte sınanır. Yeni bir
            # coğrafi bölgeyi tek başına gereksinim saymıyoruz: yalnızca
            # beklenen sonucu etkileyen tarife, fatura ve tahsilat bağlamları.
            tier_id = f"CTX_TIER_{case.offer_code}_{case.plan_tier}_{stage}"
            tags.append(tier_id)
            descriptions[tier_id] = (
                f"{case.offer_name}: {case.plan_tier} tarifesinde {stage} akisi"
            )
            important[tier_id] = stage in {"fatura", "tekrar"}

            if stage in {"fatura", "sure"}:
                if case.billing_cycle <= 1:
                    period = "ilk_donem"
                elif case.billing_cycle < 6:
                    period = "erken_donem"
                elif case.billing_cycle <= 12:
                    period = "gec_donem"
                else:
                    period = "sure_dolumu"
                bill_id = f"CTX_PERIOD_{case.offer_code}_{period}_{stage}"
                tags.append(bill_id)
                descriptions[bill_id] = (
                    f"{case.offer_name}: {period} fatura / sure durumu"
                )
                important[bill_id] = stage == "fatura"

            if case.offer_code in {"KMP03", "KMP08", "KMP09"} and stage in {
                "aktivasyon", "fatura", "uygunluk"
            }:
                pay_id = f"CTX_PAY_{case.offer_code}_{case.payment_method}_{stage}"
                tags.append(pay_id)
                descriptions[pay_id] = (
                    f"{case.offer_name}: {case.payment_method} ile {stage} akisi"
                )
                important[pay_id] = stage == "fatura"

        test_tags.append(tuple(tags))
        for tag in tags:
            counter[tag] += 1
            customers[tag].add(case.customer_id)

    retained = sorted(
        tag for tag, count in counter.items()
        if count >= 3 and len(customers[tag]) >= 2
    )
    catalog = [
        {"id": tag, "description": descriptions[tag], "critical": important[tag]}
        for tag in retained
    ]
    return catalog, test_tags


@dataclass(frozen=True)
class TestCase:
    id: str
    offer_code: str
    offer_name: str
    family: str
    title: str
    customer_id: str
    request_id: str
    line_type: str
    line_active: bool
    customer_tenure_months: int
    targeted: bool
    campaign_open: bool
    previous_use: bool
    current_contract_month: int
    billing_cycle: int
    channel: str
    request_date: str
    expected_result: str
    expected_monthly_amount_tl: float | None
    expected_data_gb: int | None
    duration_sec: int
    requirement_ids: tuple[str, ...]
    prerequisite_test_ids: tuple[str, ...]
    payment_success: bool | None = None
    remaining_quota_gb: int | None = None
    concurrent_requests: int = 1
    other_discount_active: bool = False
    plan_tier: str = "standart"
    payment_method: str = "fatura"
    customer_region: str = "marmara"
    auth_valid: bool = True
    port_in_completed: bool = True
    blacklisted: bool = False
    ownership_changed: bool = False
    customer_age: int = 30
    parent_line_active: bool = True
    roaming_enabled: bool = True


def _eligible_profile(offer: Offer, variant: int) -> tuple[str, int, bool, int]:
    """Her teklif için açıkça uygun bir müşteri örneği üretir."""
    if offer.customer_group == "yeni":
        return "faturali" if variant % 2 == 0 else "faturasiz", variant % 2, True, 0
    if offer.customer_group == "herkes":
        return "faturali" if variant % 2 == 0 else "faturasiz", 2 + variant % 19, True, 0
    if offer.customer_group == "sadakat":
        return "faturali", 6 + variant % 30, True, 0
    if offer.customer_group == "aile":
        return "faturali", 4 + variant % 24, True, 0
    if offer.customer_group == "genc":
        return "faturali" if variant % 2 == 0 else "faturasiz", 1 + variant % 12, True, 0
    if offer.commitment_required:
        return "faturali", 12 + variant % 24, True, 11 + variant % 2
    return "faturali", 2 + variant % 24, True, 0


def _benefit_amount(offer: Offer, cycle: int, cancelled: bool = False) -> float:
    """Temsili vergi hariç 200 TL temel tarife üzerinden faturayı hesaplar."""
    base = 200.0 + offer.package_fee
    if 1 <= cycle <= offer.months and not cancelled:
        base -= 200.0 * offer.monthly_discount
    return round(base, 2)


def _make_case(
    offer: Offer, family: str, variant: int, variation: int, index: int,
    expanded: bool = False,
) -> TestCase:
    line_type, tenure, targeted, contract_month = _eligible_profile(offer, variation)
    # Her onlu varyant yalnizca tek kanali degil, farkli kanallari da sinar.
    channel = ("web", "mobil", "api")[(variation + index % 10) % 3]
    customer_id = f"M_{offer.code}_{variant:03d}"
    line_active = family != "inactive_rejected"
    campaign_open = family != "window_closed"
    previous_use = family == "repeat_rejected"
    if family == "not_targeted":
        targeted = False
    if family == "commitment_ineligible":
        contract_month = 3
    if family == "tenure_rejected":
        tenure = 2
    billing_cycle = 1
    expected_amount: float | None = None
    expected_gb: int | None = None
    suffix = offer.code
    plan_tier = (
        ("standart", "premium", "aile", "genclik")[(variant + variation % 4) % 4]
        if expanded and variant > 0 else "standart"
    )
    payment_method = (
        ("fatura", "banka_karti", "dijital_cuzdan")[(variation // 7 + variant) % 3]
        if expanded and offer.package_fee > 0 else "fatura"
    )
    region = (
        ("marmara", "akdeniz", "ege", "ic_anadolu")[(variation // 13) % 4]
        if expanded else "marmara"
    )
    customer_age = (
        18 + variation % 8 if offer.customer_group == "genc"
        else 18 + variation % 50
    )
    if family == "age_ineligible":
        customer_age = 35

    # Sistemimizin kampanya başvuru penceresi: 01.01-30.09.2026.
    request_date = date(2026, 6, 15) + timedelta(days=(variation % 21) - 10)
    if not campaign_open:
        request_date = date(2026, 10, 1) + timedelta(days=variation % 15)
    if family == "boundary_last_day":
        request_date = date(2026, 9, 30)

    families: dict[str, tuple[str, str, tuple[str, ...], int]] = {
        "activation_success": (
            "Uygun musteri icin ilk aktivasyon",
            "AKTIF / tek yeni kayit",
            ("G01", "G03", "G05", "G13", f"{suffix}_E", f"{suffix}_A"),
            32,
        ),
        "inactive_rejected": (
            "Pasif hat ile kampanya basvurusu",
            "RED / kayit ve yarar olusmaz",
            ("G02", "G14", f"{suffix}_E"),
            18,
        ),
        "window_closed": (
            "Basvuru penceresi disinda aktivasyon",
            "RED / tarih kurali",
            ("G04", "G14", f"{suffix}_E"),
            17,
        ),
        "repeat_rejected": (
            "Ayni kampanya hakkinin tekrar kullanimi",
            "RED / ikinci hak yok",
            ("G06", "G14", f"{suffix}_A"),
            22,
        ),
        "first_bill": (
            "Aktif kampanyanin ilk fatura donemi",
            "BASARILI / beklenen ucret ve kota",
            ("G07", f"{suffix}_B", f"{suffix}_A"),
            45,
        ),
        "expired_benefit": (
            "Hak suresi dolduktan sonraki donem",
            "BASARILI / yarar uygulanmaz",
            ("G08", f"{suffix}_X"),
            38,
        ),
        "early_cancel": (
            "Kampanya iptalinden sonraki donem",
            "BASARILI / yarar sonlandirilir",
            ("G09", f"{suffix}_C"),
            40,
        ),
        "idempotent_request": (
            "Ayni istek kimligiyle ikinci aktivasyon istegi",
            "BASARILI / sadece bir kayit",
            ("G10", f"{suffix}_A"),
            30,
        ),
        "last_valid_bill": (
            "Kampanyanin son gecerli fatura donemi",
            "BASARILI / son kez yarar uygulanir",
            ("G11", f"{suffix}_L", f"{suffix}_B"),
            44,
        ),
        "invalid_offer_code": (
            "Gecersiz kampanya koduyla basvuru",
            "RED / teklif bulunamadi",
            ("G12", "G14"),
            14,
        ),
        "not_targeted": (
            "Hedef kitle disindaki musterinin kampanya basvurusu",
            "RED / musteri hedef kitlede degil",
            ("G15", "G14", f"{suffix}_E"),
            21,
        ),
        "commitment_ineligible": (
            "Taahhut yenileme donemi gelmeden basvuru",
            "RED / sozlesme yenileme kosulu saglanmadi",
            ("G16", "G14", f"{suffix}_E"),
            23,
        ),
        "boundary_last_day": (
            "Basvuru son gununde kampanya aktivasyonu",
            "AKTIF / bitis gunu dahildir",
            ("G17", "G03", f"{suffix}_A"),
            33,
        ),
        "concurrent_activation": (
            "Ayni anda gelen iki farkli aktivasyon istegi",
            "BASARILI / yalnizca bir aktivasyon kaydi",
            ("G18", "G05", f"{suffix}_A"),
            41,
        ),
        "payment_failure": (
            "Ucretli ek paket icin basarisiz odeme",
            "RED / odeme alinmadi ve paket tanimlanmadi",
            ("G19", "G14", f"{suffix}_E"),
            39,
        ),
        "quota_exhaustion": (
            "Internet paketi kotasi sifirlandiginda kullanim",
            "RED / kota tukendi, internet hakki kalmadi",
            ("G20", f"{suffix}_X"),
            37,
        ),
        "conflicting_discount": (
            "Baska bir indirimle catisan kampanya basvurusu",
            "RED / cakisik indirim tanimlanmaz",
            ("G21", "G14", f"{suffix}_E"),
            29,
        ),
        "tenure_rejected": (
            "Sadakat kampanyasina kidemi yetersiz musteri",
            "RED / musteri kidem kosulu saglanmiyor",
            ("G22", "G14", f"{suffix}_E"),
            28,
        ),
        "auth_expired": (
            "Oturum suresi dolduktan sonra aktivasyon talebi",
            "RED / tekrar giris yapilmadan hak tanimlanmaz",
            ("G23", "G14", f"{suffix}_E"),
            25,
        ),
        "portability_pending": (
            "Numara tasima tamamlanmadan kampanyaya basvuru",
            "RED / aktif tasinmis hat gereklidir",
            ("G24", "G14", f"{suffix}_E"),
            34,
        ),
        "prorated_upgrade": (
            "Fatura donemi ortasinda tarife yukseltme",
            "BASARILI / kalan gun ucreti dogru hesaplanir",
            ("G25", "G07", f"{suffix}_B"),
            48,
        ),
        "duplicate_billing": (
            "Ayni fatura olayi iki defa isleniyor",
            "BASARILI / ikinci tahakkuk olusmaz",
            ("G26", "G07", f"{suffix}_B"),
            44,
        ),
        "quota_reset": (
            "Yeni donemde bitmis internet kotasinin yenilenmesi",
            "BASARILI / yeni donem kotasi tanimlanir",
            ("G27", "G07", f"{suffix}_B"),
            42,
        ),
        "blacklisted": (
            "Kisitli musteri hesabiyla kampanya basvurusu",
            "RED / musteri profili izin vermiyor",
            ("G28", "G14", f"{suffix}_E"),
            23,
        ),
        "ownership_changed": (
            "Hat sahipligi degistikten sonra kampanya kontrolu",
            "BASARILI / eski kampanya hakki sonlandirilir",
            ("G29", "G09", f"{suffix}_C"),
            41,
        ),
        "second_line_ineligible": (
            "Ana hatti pasif olan kisinin aile ikinci hat basvurusu",
            "RED / bagli ana hat aktif degil",
            ("G30", "G14", f"{suffix}_E"),
            29,
        ),
        "age_ineligible": (
            "Yas kosulu uygun olmayan kisinin genc tarife basvurusu",
            "RED / genc tarife yas kosulu saglanmadi",
            ("G31", "G14", f"{suffix}_E"),
            24,
        ),
        "roaming_disabled": (
            "Yurt disi kullanim izni kapali hatta ek paket",
            "RED / roaming izni acilmadan paket kullanilamaz",
            ("G32", "G14", f"{suffix}_E"),
            28,
        ),
    }
    title, expected, rules, base_time = families[family]

    if family == "first_bill":
        expected_amount = _benefit_amount(offer, 1) if line_type == "faturali" else None
        expected_gb = offer.data_gb
    elif family == "last_valid_bill":
        billing_cycle = offer.months
        expected_amount = _benefit_amount(offer, billing_cycle) if line_type == "faturali" else None
        expected_gb = offer.data_gb
    elif family == "expired_benefit":
        billing_cycle = offer.months + 1
        expected_amount = _benefit_amount(offer, billing_cycle) if line_type == "faturali" else None
        expected_gb = 0
    elif family == "early_cancel":
        billing_cycle = min(2, offer.months)
        expected_amount = (
            _benefit_amount(offer, billing_cycle, cancelled=True)
            if line_type == "faturali" else None
        )
        expected_gb = 0
    elif family == "activation_success":
        expected_gb = offer.data_gb
    elif family == "quota_exhaustion":
        expected_gb = 0
    elif family in {"prorated_upgrade", "duplicate_billing"}:
        billing_cycle = max(1, min(offer.months, 2))
        expected_amount = (
            _benefit_amount(offer, billing_cycle)
            if line_type == "faturali" else None
        )
        if family == "prorated_upgrade" and expected_amount is not None:
            # Temsili 15/30 gun uzerinden bir aylik ek tarife farki.
            expected_amount = round(expected_amount + 30.0, 2)
    elif family == "quota_reset":
        billing_cycle = 2
        expected_gb = offer.data_gb

    # Kanal ve vaka karmaşıklığı süreye etki eder. Rastgele ilişkisiz süre yok.
    duration = base_time + {"api": 0, "web": 5, "mobil": 9}[channel] + variation % 7
    # Her varyant aynı kampanyanın kendi müşteri iş akışını kullanır.
    needs_activation = family in {
        "repeat_rejected", "first_bill", "expired_benefit", "early_cancel",
        "idempotent_request", "last_valid_bill", "quota_exhaustion",
        "prorated_upgrade", "duplicate_billing", "quota_reset",
        "ownership_changed",
    }
    offer_slot = (index % (len(OFFERS) * len(FAMILIES))) // len(FAMILIES)
    parent_id = f"T_{variant * len(OFFERS) * len(FAMILIES) + offer_slot * len(FAMILIES) + 1:04d}"
    dependencies = (parent_id,) if needs_activation else ()

    request_id = f"REQ_{offer.code}_{variant:03d}_{family}"
    if family == "idempotent_request":
        request_id = f"REQ_{offer.code}_{variant:03d}_activation_success"

    return TestCase(
        id=f"T_{index + 1:04d}",
        offer_code=offer.code,
        offer_name=offer.name,
        family=family,
        title=f"{offer.name} - {title} ({channel})",
        customer_id=customer_id,
        request_id=request_id,
        line_type=line_type,
        line_active=line_active,
        customer_tenure_months=tenure,
        targeted=targeted,
        campaign_open=campaign_open,
        previous_use=previous_use,
        current_contract_month=contract_month,
        billing_cycle=billing_cycle,
        channel=channel,
        request_date=request_date.isoformat(),
        expected_result=expected,
        expected_monthly_amount_tl=expected_amount,
        expected_data_gb=expected_gb,
        duration_sec=duration,
        requirement_ids=rules,
        prerequisite_test_ids=dependencies,
        payment_success=False if family == "payment_failure" else None,
        remaining_quota_gb=0 if family == "quota_exhaustion" else None,
        concurrent_requests=2 if family == "concurrent_activation" else 1,
        other_discount_active=family == "conflicting_discount",
        plan_tier=plan_tier,
        payment_method=payment_method,
        customer_region=region,
        auth_valid=family != "auth_expired",
        port_in_completed=family != "portability_pending",
        blacklisted=family == "blacklisted",
        ownership_changed=family == "ownership_changed",
        customer_age=customer_age,
        parent_line_active=family != "second_line_ineligible",
        roaming_enabled=family != "roaming_disabled",
    )


def generate_telecom_data(
    num_tests: int = 500,
    seed: int = 42,
    include_metadata: bool = False,
    scenario_profile: str = "standard",
):
    """En az 50 testlik, açıklanabilir ve tekrarlanabilir veri üretir.

    İlk 50 senaryo bütün 46 temel gereksinimi kapsar. Daha büyük veri kümeleri
    yeni sınır / hata senaryolarını ve tekrarlanabilen kanal-müşteri segmenti
    kapsam hedeflerini de içerir. Dolayısıyla gereksinim sayısı ölçekle artar.
    ``seed`` müşteri girdilerini, kanalı, tarihi ve süre varyasyonunu belirler;
    aynı seed ile aynı testler tekrar üretilebilir ve ilk N test daha büyük
    paketin ilk N testiyle aynıdır.

    ``dependency_matrix[parent, child] == 1``: çocuk test önce ebeveyn
    aktivasyon akışının çalıştırılmasını gerektirir.
    """
    if num_tests < 50:
        raise ValueError("Tum kural aileleri icin en az 50 test gereklidir.")
    if not isinstance(seed, int):
        raise TypeError("seed tamsayi olmalidir.")
    if scenario_profile not in {"standard", "expanded"}:
        raise ValueError("scenario_profile 'standard' veya 'expanded' olmali")
    expanded = scenario_profile == "expanded"

    rng = np.random.default_rng(seed)
    all_offers = (*OFFERS, *EXTRA_OFFERS)
    # Aynı kampanya ve varyanttaki adımlar ortak müşteri girdileri kullanır.
    group_variations: dict[tuple[int, int], int] = {}
    cases = []
    for index in range(num_tests):
        variant, rem = divmod(index, len(OFFERS) * len(FAMILIES))
        offer_index, family_index = divmod(rem, len(FAMILIES))
        offer = (
            all_offers[(variant * len(OFFERS) + offer_index) % len(all_offers)]
            if expanded and variant > 0
            else OFFERS[offer_index]
        )
        group = (variant, offer_index)
        if group not in group_variations:
            group_variations[group] = int(rng.integers(0, 2_000_000_000))
        family = _families_for_variant(
            offer, variant, expanded=expanded
        )[family_index]
        cases.append(
            _make_case(
                offer, family, variant,
                group_variations[group], index, expanded=expanded,
            )
        )

    # Kataloğu havuzda görülen ek kampanyalar, kurallar ve bağlamlarla genişlet.
    observed_offer_codes = {c.offer_code for c in cases}
    observed_extra_offers = tuple(
        offer for offer in EXTRA_OFFERS if offer.code in observed_offer_codes
    )
    base_catalog = requirement_catalog(observed_extra_offers)
    observed_rules = {r for case in cases for r in case.requirement_ids}
    extra_catalog = [
        {"id": code, "description": desc, "critical": critical}
        for code, desc, critical in EXTENDED_RULES if code in observed_rules
    ]
    context_catalog, tags_by_case = _context_catalog(cases, expanded=expanded)
    catalog = base_catalog + extra_catalog + context_catalog
    features = [f"{r['id']}: {r['description']}" for r in catalog]
    feature_index = {r["id"]: i for i, r in enumerate(catalog)}
    test_index = {case.id: i for i, case in enumerate(cases)}

    # Kapsamda satır test, sütun hedeftir. Bağımlılıkta satır ön koşul, sütun bağlı testtir.
    coverage = np.zeros((num_tests, len(catalog)), dtype=np.int8)
    dependencies = np.zeros((num_tests, num_tests), dtype=np.int8)
    for i, case in enumerate(cases):
        for requirement_id in case.requirement_ids:
            coverage[i, feature_index[requirement_id]] = 1
        for tag in tags_by_case[i]:
            if tag in feature_index:
                coverage[i, feature_index[tag]] = 1
        for parent_id in case.prerequisite_test_ids:
            dependencies[test_index[parent_id], i] = 1

    durations = np.asarray([case.duration_sec for case in cases], dtype=int)
    names = [case.title for case in cases]
    result = (coverage, durations, names, features, dependencies)

    # Notebook, tabloları ve grafikleri oluşturmak için senaryo ayrıntılarını ister.
    if include_metadata:
        metadata = {
            "synthetic": True,
            "dataset_name": "Kurgusal telekom kampanya ve taahhut sistemi",
            "scenario_profile": scenario_profile,
            "seed": seed,
            "offers": [asdict(o) for o in (*OFFERS, *observed_extra_offers)],
            "requirements": catalog,
            "core_rule_count": len(base_catalog),
            "additional_rule_count": len(extra_catalog),
            "context_rule_count": len(context_catalog),
            "family_counts": dict(Counter(c.family for c in cases)),
            "critical_feature_indices": [i for i, r in enumerate(catalog) if r["critical"]],
            "test_cases": [asdict(c) for c in cases],
        }
        return (*result, metadata)
    return result


def validate_telecom_data(num_tests: int = 500) -> dict[str, int]:
    """Boyutları, referansları ve test kural kapsamını doğrular."""
    coverage, durations, names, features, dependencies, metadata = generate_telecom_data(
        num_tests, include_metadata=True
    )
    assert coverage.shape[0] == num_tests and coverage.shape[1] >= 46
    assert durations.shape == (num_tests,)
    assert len(names) == num_tests and len(features) == coverage.shape[1]
    assert dependencies.shape == (num_tests, num_tests)
    assert np.all(coverage.sum(axis=0) >= 1), "Kapsanmayan kural var"
    assert np.all(durations > 0)
    assert np.all(np.tril(dependencies) == 0), "Bagimliliklar dongusuz/ileri yonlu olmali"
    assert len(set(c["id"] for c in metadata["test_cases"])) == num_tests
    return {
        "tests": num_tests,
        "requirements": len(features),
        "critical_requirements": len(metadata["critical_feature_indices"]),
        "additional_business_rules": metadata["additional_rule_count"],
        "context_requirements": metadata["context_rule_count"],
        "dependency_edges": int(dependencies.sum()),
        "total_duration_sec": int(durations.sum()),
    }


if __name__ == "__main__":
    for size in (50, 100, 250, 500, 1000):
        print(validate_telecom_data(size))
    _, _, _, _, _, details = generate_telecom_data(50, include_metadata=True)
    print("\nIlk 3 ornek test:")
    for item in details["test_cases"][:3]:
        print(item)
