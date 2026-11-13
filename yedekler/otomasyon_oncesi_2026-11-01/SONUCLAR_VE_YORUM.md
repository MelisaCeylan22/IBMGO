# IBMGO: deney ozeti ve sunum notlari

## 1. Ne yaptik?

Kurgusal telekom regresyon test havuzlari (50, 100, 250, 500, 1000 test)
icin uc sert kosul korunarak test paketi secilir:

1. Kritik gereksinimlerin tamamini (%100) kapsa.
2. Genel gereksinimlerin en az %90'ini kapsa.
3. Secilen tum testlerin on kosul testleri de secili olsun.

Uygun cozumlerde test kosum suresi, test sayisi ve gereksiz kapsam tekrari
birlikte minimize edilir. Hata tespit matrisi olmadigi icin APFD ve gercek
hata yakalama hizi **olculmez**. Secilmis testler ayrica erken kritik
gereksinim kapsami icin bagimliliklara uyarak siralanir.

## 2. IBMGO'nin teknik katkisi

- Ozgun yazar MGO kodunun (2022) TSM, MH, BMH ve MSF hareketleri, katsayi
  aileleri ve suru ortalamasi incelendi. Ikili test secimine yaklastirilarak
  aktarildi; **ozgun surekli MGO'nun birebir kopyasi degildir**.
- 2024 PLOS ONE calismasindaki ICMIC kaotik baslangic, dogrusal olmayan
  kontrol, spiral arama ve komsuluk fikrinden yararlanildi. Basarili olmayan
  sigmoid/spiral secenekleri de ablasyon deneyi olarak saklandi.
- V-sekilli bit degistirme olasiligini sinirlandirma, tekli ve coklu yerel
  komsuluk ve hareket ailelerini karistirma teknikleri ayrica denendi.
- Gelistirme veri tohumlarina (42/53) gore **oncesinden sabitlenen** politika:

| Test buyuklugu | Kullanilan IBMGO aramasi |
| --- | --- |
| 50, 100, 250 | Adaptif V-donusum ve tekli komsuluk |
| 500 | Referans MGO'ya yakin hareketler, ICMIC, dogrusal olmayan kontrol ve coklu komsuluk |
| 1000 | Referans ve onceki hareketlerin %50 karisimi, ICMIC ve tekli komsuluk |

Bu esikler **deneyseldir**. 71 veri tohumu, parametreler/esikler sabitlendikten
sonra bagimsiz kontrol icin kullanilmistir. GA ve IBMGO her kosumda **1212**
aday fitness degerlendirmesine sahiptir. Greedy deterministik tek gecistir.

## 3. Bagimsiz veri kumesi: seed=71

Asagidaki degerler, **uc bagimsiz optimizasyon tohumunun (42,43,44)
ortalamasidir**. Greedy deterministiktir. Fitness dusukse iyidir, tasarruf
yuksekse iyidir.

| Test | IBMGO fitness | GA fitness | Greedy fitness | IBMGO sure tasarrufu | GA sure tasarrufu | Greedy sure tasarrufu |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.526179 | 0.526179 | 0.531544 | %43.63 | %43.63 | %42.89 |
| 100 | 0.461076 | 0.461076 | 0.469019 | %52.37 | %52.37 | %52.00 |
| 250 | 0.290823 | 0.290406 | 0.324025 | %74.62 | %74.68 | %72.76 |
| 500 | 0.262312 | 0.262894 | 0.297351 | %79.81 | %79.77 | %77.66 |
| 1000 | 0.214914 | 0.213738 | 0.240027 | %86.81 | %86.76 | %85.73 |

**Dogrulanan sonuc:** IBMGO tum bu boyutlarda Greedy'den daha dusuk fitness
buldu; GA ile benzer/rekabetci sonuclar verdi. 500 testte GA'dan az farkla
daha dusuk fitness buldu; 250 ve 1000 testte GA fitness'ta az farkla ondeydi.
Ozellikle farklar kucuk oldugu icin istatistiksel ustunluk iddiasi yoktur.
Butun karsilastirmalarda kritik kapsam %100, genel kapsam en az %90 ve
on kosul ihlali sifirdir.

**Sinirlar:** Sentetik testler ve sureler, gercek operator loglarindan
gelmiyor. Farkli olasi veri dagilimlarinda sonuclar degisebilir. Gelistirme
politikasinin baska alanlara da aynen uyacagi iddia edilmez.

**Onceliklendirme ayri bir katki sagladi:** 42 veri tohumu ile 1000 testlik
pakette, IBMGO'nin sectigi **ayni testler** rastgele ama on kosullara uygun
siralandiginda ilk %25 kosum suresinde kritik kapsam ortalamasi **%23.8**;
akilli sira ile **%39.3** oldu. Fark **15.5 yuzde puan**. Bu, gercek
hatalari %15.5 daha erken bulduk anlamina **gelmez**; kritik gereksinim
kapsamini daha erken elde ettigimiz anlamina gelir.

## 4. Sunum icin aciklama

> "Bu projede BMGO'yu telekom regresyon test secimine uyarladik. Ozgun MGO
> hareketlerini inceleyip baslangic ve komsuluk aramasini problem boyutuna
> gore yonettik. Tek hedefimiz test sayisini azaltmak degil; kritik kurallarin
> tamamina, genel kurallarin en az %90'ina ve tum on kosullara uyarken zaman
> tasarrufu saglamakti. Bagimsiz veri deneyinde IBMGO, Greedy'den daha iyi
> fitness elde etti, GA ile rekabetci performans gosterdi. Ayrica secilen
> testleri kritik kurallari erken kapsayacak sekilde siraladik."

## 5. Kayitlar ve calistirma

- `sonuclar/nihai_bmgo/algoritma_ozet.png`: Gelistirme verisinde 3 tekrarli
  fitness, sure tasarrufu ve ilk %25 kritik kapsam tablosu.
- `sonuclar/nihai_bmgo/algoritma_karsilastirma.png`: Yontemlerin sure tasarrufu.
- `sonuclar/nihai_bmgo/bmgo_iyilestirme_yakinsama.png`: 1000 testte es aday
  degerlendirme butcesiyle yakinsama.
- `sonuclar/nihai_bmgo/onceliklendirme_kritik_kapsam.png`: Ayni testler
  uzerinde kritik kapsamin onceliklendirilmis/rastgele sira karsilastirmasi.
- `sonuclar/bagimsiz_dogrulama_seed71/ablasyon_veri71.csv` ve `.png`:
  Bagimsiz veri kumesindeki gercek karsilastirma; sunumda ayri gosterilmeli.

Windows/VS Code terminalinde:

```powershell
python -m pip install numpy matplotlib
python -B -m unittest test_project.py -v
python main.py --runs 3 --data-seed 42
python priority_analysis.py --random-orders 100 --data-seed 42
python ablation.py --sizes 50 100 250 500 1000 --runs 3 --data-seed 71 --output-dir sonuclar/bagimsiz_dogrulama_seed71 --variants BMGO_boyuta_adaptif BMGO_artı
```

## 6. Akademik kaynaklar

1. Abdollahzadeh ve ark. (2022), *Mountain Gazelle Optimizer: A new
   Nature-inspired Metaheuristic Algorithm for Global Optimization Problems*,
   DOI 10.1016/j.advengsoft.2022.103282.
   [Yazar MGO kodu](https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer).
2. Li, Geng, Sheng (2024), *An improved mountain gazelle optimizer based on
   chaotic map and spiral disturbance for medical feature selection*,
   DOI 10.1371/journal.pone.0307288,
   [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307288).
