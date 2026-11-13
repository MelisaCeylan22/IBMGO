# IBMGO sağlamlık ve tekrar edilebilirlik analizi

**Deney tasarımı (önceden sabitlendi):** IBMGO'nin boyuta uyarlanan ayarları değiştirilmeden test edilmiştir. GA da 1212 fitness değerlendirmesi kullanır. Greedy deterministiktir; aynı veri kümesinde tekrar çalıştırılmamıştır.

- Sentetik telekom veri seti başlangıçları: **83, 97, 109** (3 farklı set).
- Her set/test büyüklüğünde IBMGO ve GA için **10** ayrı optimizer başlangıcı.
- Her test büyüklüğünde **30** eşleşmiş IBMGO/GA deneyi; Greedy için **3** ayrı veri kümesi sonucu.
- Kritik gereksinim kapsamı %100, genel gereksinim kapsamı en az %90, ön koşul ihlali sıfır — tüm kayıtlarda doğrulandı.
- Değerler sentetiktir. Gerçek hata tespit matrisi olmadığı için APFD hesaplanmamıştır.

## Ana tablo

| Test | IBMGO fitness | GA fitness | Greedy fitness | IBMGO tasarruf | GA tasarruf | Greedy tasarruf |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.5265 ± 0.0016 | 0.5265 ± 0.0016 | 0.5306 | %43.54 | %43.54 | %42.94 |
| 100 | 0.4601 ± 0.0038 | 0.4589 ± 0.0022 | 0.4725 | %52.38 | %52.44 | %51.49 |
| 250 | 0.3006 ± 0.0115 | 0.3067 ± 0.0116 | 0.3244 | %73.70 | %73.36 | %72.26 |
| 500 | 0.2545 ± 0.0088 | 0.2587 ± 0.0086 | 0.2836 | %80.62 | %80.32 | %78.90 |
| 1000 | 0.2188 ± 0.0031 | 0.2171 ± 0.0028 | 0.2494 | %86.44 | %86.44 | %84.98 |

*± işareti IBMGO ve GA için 30 optimizer/veri koşumunun örnek standart sapmasıdır. Greedy ortalaması yalnız üç veri kümesi üzerindendir.*

## IBMGO − GA eşleşmiş farkları

Fitness düşük daha iyi; **negatif fark IBMGO lehine**. Güven aralıkları veri kümesi ve optimizer başlangıcı düzeyinde hiyerarşik bootstrap ile hesaplanan betimleyici aralıklardır; yalnız üç veri kümesi bulunduğundan kesin genelleme veya anlamlılık kanıtı olarak yorumlanmamalıdır.

| Test | IBMGO kazanma | Eşit | GA kazanma | Ort. fark | %95 bootstrap aralığı |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0 | 30 | 0 | +0.00000 | [+0.00000, +0.00000] |
| 100 | 4 | 21 | 5 | +0.00117 | [-0.00019, +0.00413] |
| 250 | 24 | 0 | 6 | -0.00612 | [-0.00855, -0.00378] |
| 500 | 23 | 0 | 7 | -0.00421 | [-0.00659, -0.00181] |
| 1000 | 12 | 0 | 18 | +0.00168 | [-0.00054, +0.00414] |

## Sunumda güvenle söylenebilecekler

- IBMGO, bu sentetik telekom probleminde her tekrarda geçerli ve kısıtlara uyan test paketi oluşturmuştur.
- Farklı başlangıçlarda sonuçların ne kadar değiştiği ve hangi veri kümelerinde hangi yöntemin öne geçtiği artık raporlanabilir; tek bir iyi koşuma dayanmıyoruz.
- Greedy'nin gerçek çalışma zamanı ile IBMGO/GA'nın çalışma zamanı aynı donanımda ölçülse bile tek geçişli Greedy'nin aday bütçesi eşit değildir: kalite ve hesaplama maliyeti farklı eksenlerde yorumlanmalıdır.
- Eşleşmiş optimizer seed'leri ortak veri koşullarını kontrol eder, ancak IBMGO ve GA'nın aynı rassal sayı akışını kullandığı anlamına gelmez.

## IBMGO'nin çalışma mantığı (mekanizma açıklaması)

- 50–250 test: farklı başlangıç çözümleri, ikili dönüşüm ve yerel komşuluk araması birlikte kullanılır.
- 500 test: referans MGO hareketleri, kaotik başlangıç, doğrusal olmayan kontrol ve daha geniş komşuluk araması devrededir.
- 1000 test: referans ve uyarlanmış hareketler karıştırılır; komşuluk araması daha küçük değişikliklere döner.
- Amaç, geniş aday uzayını araştırırken seçilmiş iyi test paketlerinin çevresinde de arama yapmaktır. Bu bir tasarım gerekçesidir; tek başına hangi bileşenin fayda sağladığını kanıtlamaz. Nedensel iddia için ayrı ablasyon sonuçları gerekir.

## Sınırlılıklar

- Yalnız üç sentetik veri kümesi: gerçek operatör üretim verisine genellenemez. Verinin üretim kuralları optimize edilmemiştir.
- 1212 fitness değerlendirmesi eşit olsa da algoritmaların duvar saati süresi ve aday başına maliyeti farklıdır.
- Farklı test büyüklükleri farklı gereksinim ve test havuzlarıdır; fitness değerlerini *boyutlar arasında* doğrudan sıralamak yerine her boyut içinde algoritmalar arası karşılaştırma yapılmalıdır.
- Test süreleri toplamı optimizasyon hedefidir; CI/CD boru hattındaki gerçek yürütme süresi sahada ölçülmemiştir.
- Hata erken yakalama performansı için doğrulanmış arıza-test matrisi gerekir; önceliklendirme göstergeleri gerçek APFD yerine geçmez.
