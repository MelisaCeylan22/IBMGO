# Telekom regresyon test seçimi ve önceliklendirme

Bu çalışma, **tamamen kurgusal** bir telekom kampanya / taahhüt sisteminin
regresyon testlerini azaltmak için dört yöntemi karşılaştırır:

1. **Greedy:** Açgözlü kapsam onarımı ve gereksiz testlerin ayıklanması.
2. **GA:** İkili genetik algoritma (seçim, çaprazlama, mutasyon, elitizm).
3. **BMGO:** TSM, MH, BMH, MSF hareketlerini kullanan ikili MGO uyarlaması.
4. **IBMGO (ana önerilen yöntem):** Problem boyutuna göre MGO hareketini
   ayarlayan, adaptif V-tipi ikili dönüşüm ve komşuluk araması kullanan
   deneysel BMGO. Orta/büyük paketlerde referans MGO yazar kodundaki katsayı
   ailelerine daha yakın TSM/MH/BMH/MSF, ICMIC kaotik başlangıç ve doğrusal
   olmayan kontrol devreye girer. **Özgün MGO veya Li ve ark. (2024) BIMGO
   algoritmasının birebir kopyası değildir; probleme özgü bir uyarlamadır.**

İlk BMGO, geliştirmenin etkisini incelemek amacıyla ek karşılaştırmadır.
Sunumun ana sonucu IBMGO'nin GA ve Greedy ile karşılaştırılmasıdır.

## Kurulum ve çalıştırma

Python 3.10 ve üzeri önerilir.

```bash
python -m pip install numpy matplotlib
python -B -m unittest test_project.py -v
python main.py
python priority_analysis.py --random-orders 100
python ablation.py --sizes 250 500 --runs 2
```

`main.py` 50, 100, 250, 500 ve 1000 testlik paketleri her stokastik yöntem
için 3 ayrı başlangıçla çalıştırır. `priority_analysis.py` ilk seed için
seçilmiş aynı testleri farklı sıralarla karşılaştırır. `ablation.py` ise
her iyileştirmeyi ayrı ayrı açıp kapatarak denetler.
Deney sayısını artırmak ve başka kurgusal
senaryoları üretmek için:

```bash
python main.py --runs 10 --profile expanded --data-seed 53 --output-dir sonuclar/dogrulama53
python priority_analysis.py --data-seed 53 --output-dir sonuclar/dogrulama53
```

Eski veri profiliyle geriye dönük karşılaştırma:

```bash
python main.py --runs 3 --profile standard --data-seed 42 --output-dir sonuclar/standart
```

**Dikkat:** Yeni ana deney çıktıları varsayılan olarak
`sonuclar/nihai_bmgo` dizinine gider; eski deney dosyaları korunur.
Aynı çıkış klasörüyle yeniden çalıştırırsanız o klasörde aynı isimli
CSV/PNG dosyaları güncellenir. Önceden kalan başka isimli grafikler silinmez.

## Telekom senaryoları

Genişletilmiş veri üretiminde **dokuz** kurgusal teklif / paket ailesi bulunur.
İlk 50 senaryodaki beş temel teklif şunlardır:

- 12 ay indirim
- yeni müşteri internet hediyesi
- ücretli ek internet paketi
- sadakat indirimi
- taahhüt yenileme

50 testin ötesindeki yeni senaryo grupları dört ayrı teklife daha yayılır:

- numara taşıma sonrası 15 GB paketi
- aileye bağlı ikinci hatta %10 indirim
- gençlere özel 20 GB tarifesi ve yaş denetimi
- yurt dışı 3 GB paketi ve uluslararası dolaşım izni

İş akışları; başarı, uygunsuz müşteri, kapalı başvuru tarihi, tekrar kullanım,
fatura kontrolü, son dönem, iptal, idempotent istek, eşzamanlı aktivasyon,
ödeme başarısızlığı ve kota tükenmesini içerir. Genişletilmiş profilde ayrıca
oturum süresi dolması, numara taşıma beklenmesi, kısıtlı müşteri, ara dönem
tarife değişikliği, mükerrer faturalama, kota yenilenmesi ve hat sahipliği
değişikliği bulunur. İkinci hat için ana hattın aktif olması, genç tarifesinde
yaş sınırı ve yurt dışı paketinde dolaşım izni, ayrı başarısızlık senaryolarıyla
sınanır. Ücretli yeni paketlere ödeme başarısızlığı kontrolü de uygulanır.

**Önemli:** Birbirinden farklı müşteri, tarife, tahsilat ve fatura dönemi
bağlamları yeni test kapsam hedefleri üretebilir. Bunlar doğrulanmış gerçek
operatör kuralları değil, örnek iş kurallarıdır. Bölge/şehir tek başına
bir gereksinim sayılmaz.

## Optimizasyon hedefi ve kısıtlar

- Kritik gereksinim kapsamı **%100** olmalıdır.
- Toplam gereksinim kapsamı **en az %90** olmalıdır.
- Seçilen testlerin tüm ön koşul testleri de seçilmiş olmalıdır.
- Uygun çözümler arasında koşum süresi, test sayısı ve tekrarlı kapsam azaltılır.

Kısıtlar sağlanmadığında fitness büyük ceza alır. Uygun çözümlerin fitness
fonksiyonu:

```text
F(X) = 0.4 × (seçilen süre / toplam süre)
     + 0.4 × (seçilen test / toplam test)
     + 0.2 × (tekrarlı kapsam / toplam kapsam isabeti)
```

**Düşük fitness daha iyi**. Farklı büyüklüklerdeki mutlak fitness değerleri
aynı problem gibi doğrudan karşılaştırılmamalıdır.

## Karşılaştırma adaleti

Greedy tek deterministik çalışma yapar; diğerleri ayrı rastgele başlangıçlarla
çalışır. Her üç popülasyon algoritması aynı uygunluk onarımını ve amaç
fonksiyonunu kullanır. Böylece değişen temel bileşen arama mekanizması olur.

12 bireylik popülasyonda her yöntem toplam **1212 fitness** değerlendirmesi
yapar:

- BMGO: 12 başlangıç + 25 iterasyon × 48 aday.
- IBMGO: 12 başlangıç + 20 iterasyon × (48 MGO adayı + 12 komşu).
- GA: 12 başlangıç + 100 nesil × 12 aday.

Eşit değerlendirme sayısı, eşit çalışma süresi anlamına gelmez. Her yöntem
ayrı ayrı gerçek çalışma süresiyle de raporlanır.

## Önceliklendirme ve APFD ayrımı

Seçilmiş testler, bağımlılıklar korunarak yeni kritik gereksinimlerin
koşum süresine oranı yüksek olanlardan başlanarak sıraya konur.
`ilk_ceyrek_kritik_kapsam_yuzde` metriği, sıradaki testlerin ilk %25'inin
kapsadığı kritik gereksinim oranını gösterir.

**Bu metrik APFD değildir.** Veri üreticisi doğrulanmış hata tespit matrisi
üretmez. Bu yüzden gerçek hata keşif hızı veya APFD sonucu iddia edilmez;
`APFD=VERI_YOK` korunur. Ayrıca sentetik koşum süreleri, gerçek CI/CD üzerinde
ölçülmüş sürelerden farklıdır.

`priority_analysis.py` seçilmiş **aynı testleri** koruyarak, kritik kapsamın
ne kadar çabuk tamamlandığını iki sıralamayla karşılaştırır: önceliklendirilmiş
sıra ve rastgele seçilmiş, ancak bütün ön koşulları koruyan 100 sıra. Böylece
test seçme başarısı ile test sıralama başarısı birbirine karıştırılmaz.
`kritik_kapsam_auc` (0–1 arası) kritik kural kapsamının toplam koşum zamanı
boyunca ne kadar erken biriktiğini özetler; **APFD değildir**.

## Çıktılar

Ana deney sonucunda `sonuclar/nihai_bmgo` klasörüne otomatik olarak:

- `algoritma_karsilastirma.csv`: Her tohum için ayrıntılı deney sonuçları.
- `algoritma_ozet.csv`: Her paket/yöntem için ortalama ve standart sapma.
- `algoritma_ozet.png`: Sunum için özet karşılaştırma tablosu.
- `algoritma_karsilastirma.png`: Süre tasarrufu karşılaştırması.
- `algoritma_fitness_karsilastirma.png`: Fitness karşılaştırması.
- `bmgo_iyilestirme_yakinsama.png`: 1000 testte eşit değerlendirme bütçesi.
- `onceliklendirme_analizi.csv`: Kritik kapsamın erken kazanımı ve 100 rastgele
  bağımlılık-geçerli sıraya göre karşılaştırması.
- `onceliklendirme_kritik_kapsam.png`: Kritik kapsamın tamamlanan koşum süresine
  göre birikmesi; erken geri bildirim iddiası için ayrı grafik.
- `telekom_50_ornek_senaryo.csv`: Örnek senaryo açıklamaları.
- `telekom_<N>_secili_testler.csv`: Ana IBMGO tarafından seçilmiş ve sıralanmış testler.
- `telekom_<N>_bmgo_ilk_secili_testler.csv`: İlk BMGO'nun kontrol amaçlı seçtiği testler.

`ablation.py` ayrıca `sonuclar/ablasyon_veri42.csv` ve `.png` üretir;
ayrı `--output-dir` ve `--data-seed` ile doğrulama deneyleri saklanabilir.

### Ablasyon: Yarar sağlayan ve sağlamayan teknikler

Deney varyantları: ilk BMGO; önceki adaptif BMGO+; referans MGO hareketleri;
kaotik başlangıç; doğrusal olmayan kontrol; ayrıca spiral hareket ve sigmoid
ikili dönüşüm; hareketleri karıştırma, ince V-dönüşüm ve çoklu komşuluk araması.
Spiral ve sigmoid seçenekleri **denenmiştir** ancak varsayılan IBMGO'ye
eklenmemiştir. Özellikle sigmoid dönüşüm bazı veri kümelerinde ciddi performans
düşüşü vermiştir. Aynı aday bütçesiyle kıyas yapılır. Deneyin tek amacı
başarılı görünen değişimleri raporlamak değil, başarısız denemeleri de
belgelendirmektir.

### Boyuta duyarlı IBMGO politikası

| Test sayısı | Uygulanan hareket/yerel arama |
| --- | --- |
| 1–250 | Adaptif V-dönüşüm, çeşitli başlangıç ve tekli swap komşuluğu |
| 251–750 | Referansa yakın hareketler, ICMIC, doğrusal olmayan kontrol ve çoklu komşuluk |
| 751+ | Referans/ilk hareketlerin %50 karışımı, ICMIC, doğrusal olmayan kontrol ve tekli komşuluk |

Bu aralıklar **42 ve 53 veri tohumlarında yapılan keşifsel çalışmadan sonra
belirlenmiştir**. Evrensel bir biyolojik/istatistiksel eşik iddiası değildir.
`--data-seed 71`, bu kuralı belirledikten sonra ilk kez kullanılan bağımsız
doğrulama verisidir. Bu aşamanın sonuçlarına bakılarak aralıklar değiştirilmez.

## Bilimsel çalışma notu

Başlangıç parametreleri üzerinde yapılan ilk denemeler **keşifsel** niteliğe
sahiptir. Bir algoritmayı kazandırmak için veri veya karşılaştırma ayarları
sonradan değiştirilmemelidir. Bulguların dayanıklılığını denetlemek için
farklı `--data-seed` değerleri ve çoklu bağımsız koşumlar kullanılmalıdır.
IBMGO'yi özgün, sürekli MGO'dan ve ilk ikili prototipten kavramsal olarak
ayırmak gerekir. `--data-seed 42` ve `53` **geliştirme/ayar seçimi**, `71` ise
**bağımsız doğrulama** için kullanılmıştır. İki algoritma arasındaki küçük
ortalama farkları, yeterli koşum ve istatistiksel denetim olmadan üstünlük
kanıtı olarak sunulmamalıdır.

### Kaynaklar

- Abdollahzadeh ve ark. (2022), *Mountain Gazelle Optimizer: A new
  Nature-inspired Metaheuristic Algorithm for Global Optimization Problems*.
  DOI: [10.1016/j.advengsoft.2022.103282](https://doi.org/10.1016/j.advengsoft.2022.103282).
  [Özgün yazar kodu](https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer).
- Li, Geng ve Sheng (2024), *An improved mountain gazelle optimizer based on
  chaotic map and spiral disturbance for medical feature selection*.
  DOI: [10.1371/journal.pone.0307288](https://doi.org/10.1371/journal.pone.0307288).

Bu kaynaklar yöntem seçim ve iyileştirme fikirlerini destekler; bizim kurgusal
telekom veri kümesindeki başarıyı bağımsız olarak doğrulamaz.
