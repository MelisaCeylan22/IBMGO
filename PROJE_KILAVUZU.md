# IBMGO projesi ve sunum kullanımı

Ana sunum `IBMGO_Sunum.ipynb` dosyasıdır. Python çekirdeğiyle proje klasöründe açıp hücreleri sırayla çalıştırın. Açılış akışı optimizasyon başlatmaz. Kayıtlı ana dayanıklılık sonuçlarını okur.

## Kurulum

```powershell
python -m pip install numpy pandas matplotlib ipywidgets nbformat nbclient ipykernel
python -B -m unittest discover -s kod/testler -v
```

## Üç ayrı deney

| Deney | Boyutlar | Veri tohumları | Algoritma tekrarları | Kaynak |
|---|---|---|---|---|
| Ana dayanıklılık arşivi | 50, 100, 250, 500, 1000 | 83, 97, 109 | IBMGO ve GA için 42–51, Greedy bir kez | `sonuclar/dayaniklilik_83`, `dayaniklilik_97`, `dayaniklilik_109` |
| Hızlı canlı demo | Varsayılan 50 | 83 | 42, Greedy bir kez | Yeni çalışma klasörü |
| Ek ölçek deneyi | 500, 1000, 5000 | 83, 42, 41, 97, 109 | Eski notebook ayarları, tek algoritma tohumu | `IBMGO_Ek_Olcek_Deneyi.ipynb` |

Ana arşiv 315 ham kayıt içerir. IBMGO ve GA boyut başına 30, Greedy 3 kayıtla özetlenir. Ham dosyaların adı `ablasyon_veri<tohum>.csv` olsa da yalnız IBMGO, GA ve Greedy içerir. İlk BMGO ve tüm bileşenlerin tam ablasyonu bu dosyalarda yoktur.

Ek ölçek notebook'u önceki kaynak akışını korur, büyük deneyi açıkça başlatabilir. Bu pakette o ölçek deneyine ait tamamlanmış ve ayar dosyasıyla doğrulanabilen ham sonuçlar yoktur. Ana tabloyla birleştirilmez.

## Canlı gösterim

1. Ana notebook'ta hazır mod seçin. Test boyutları, tohumlar, kapsam, ağırlıklar, popülasyon, iterasyon ve yerel aday sayısını panelden değiştirin.
2. Veri alanında bir havuz, test ve hedef seçin. Süreyi, kritik hedef işaretini veya kapsam hücresini değiştirip **Veri değişikliğini uygula** düğmesine basın. **Orijinal veriye dön** seçilen havuzu sıfırlar.
3. **Deneyi çalıştır / Sonuçları güncelle** bütün yöntemleri aynı kopyayla çalıştırır. İlerleme tamamlanmış koşum sayısıdır. Her yöntem kendi araması sürerken sayaç o yöntem bitince ilerler.
4. Sonuç ekranındaki boyut ve koşum filtreleri yeniden arama yapmaz. Ham tablo ve ortalama özeti ayrı gösterilir.
5. **Kayıtlı ana deneyi göster** arşive döner. Canlı panel ayarları arşiv sonuçlarına uygulanmaz. Sonuç kaynağı ve kullanılan ayarlar ekranda kalır.

Widget kurulumu yoksa notebook'un sonundaki `CALISTIR` hücresi aynı motoru kullanır. Ayarlar `AYARLAR` üzerinden yönetilir. Veri kopyaları `deney.pool`, `deney.set_pool`, `deney.reset_pool` ile düzenlenir. Widgetsız kullanımda eski tabloyu yeniden kullanmadan önce `deney.stale` kontrol edilir.

## Problem ve amaç

Telekom teklifleri sentetiktir. Kampanya aktivasyonu, fatura, tekrar istek, kapalı başvuru tarihi, pasif hat, numara taşıma, ikinci hat, genç tarifesi ve dolaşım izni gibi kurallar kullanılır. Gerçek operatör sistemine bağlı değildir.

Her test 0/1 seçilir. Kritik kapsam %100, genel kapsam en az %90, ön koşul ihlali sıfırdır. Ön koşul grafiği sıralı ve döngüsüzdür, ebeveyn indeksi çocuktan küçüktür. Gerekli kapsamın olanaksızlığı arama öncesi bildirilir.

Uygun paketlerde fitness, süre oranı, test sayısı oranı, tekrar oranı ve isteğe bağlı nadir hedef kaybının ağırlıklı ortalamasıdır. Ana ağırlıklar 0,40 / 0,40 / 0,20 / 0. Bütün ağırlıklar toplamlarına bölünür. Düşük fitness daha iyidir.

MGO hareketleri ikili test seçimine uyarlanmıştır. IBMGO çeşitli başlangıç, onarım, yerel arama ve mevcut boyuta duyarlı politikayı kullanır. Politika 42 ve 53 veri tohumlarıyla keşifsel olarak belirlenmiştir. Bu düzenleme üreticiyi, onarımı veya arama politikasını değiştirmez. Özgün MGO veya makaledeki yöntemin birebir kopyası iddiası yoktur.

## Bütçe ve yakınsama

Standart IBMGO planı `12 + 20 × (4 × 12 + 12) = 1212`, GA planı `12 + 100 × 12 = 1212` çağrıdır. İleri ayarlardaki spiral adaylar da bütçeye katılır. En yakın GA nesil sayısı otomatik seçilir, bütçe farkı gösterilir. Gerçek çağrı sayıları izden okunur. Greedy tek deterministik yöntemdir. Çağrı sayısı, benzersiz paket veya aynı çalışma süresi anlamına gelmez.

Algoritmalardaki isteğe bağlı `trace={}` parametresi mevcut üçlü dönüşü değiştirmez. Başlangıç, her amaç çağrısı, yerel/spiral arama ve adım sonu eğrileri kaydedilir. İterasyon/nesil grafikleri başlangıç popülasyonunu içerir. Ortak grafik gerçek kümülatif değerlendirmeyi kullanır. Aynı veri ve boyuttaki diğer tekrarlar silik gösterilir.

Nihai en iyiye ilk erişim o koşumun nihai değerine ilk erişimdir. Eşitlik toleransı rtol=1e-10, atol=1e-12. Küresel optimum veya kesin yakınsama adımı değildir. Arşivde iz olmadığından geçmiş üretilmez.

## Yeni sonuçların kaydı

Her yeni çalışma `sonuclar/sunum_calistirmalari/<UTC_zamanı>_<kimlik>` altında saklanır.

- `ayarlar.json` ayarları, çözülmüş politika seçeneklerini, planlanan bütçeleri, girdi ve kaynak kod özetlerini, sürümleri ve tamamlanma durumunu içerir.
- `girdi_<boyut>_<tohum>.npz` gerçek kullanılan kapsamı, süreleri, bağımlılıkları ve kritik hedefleri içerir.
- `veri_<boyut>_<tohum>.json` senaryo metinleri ve kaynak metadata içerir. Düzenlenmiş süre/kapsam/kritiklik için NPZ esas alınır.
- `ham_sonuclar.csv`, `ozet.csv`, `izler_ve_secimler.json` aynı hesaplamanın çıktılarıdır.

Önbellekten optimizasyon sonucu döndürülmez. Girdi parmak izi veri dizilerini ve bütün deney ayarlarını kapsar. Yeni hesaplama başarılı olmadan eski sonuç yenisiyle değiştirilmez.

## Eksik arşivler ve sınırlar

`sonuclar/nihai_bmgo`, `sonuclar/bagimsiz_dogrulama_seed71`, seed 42 önceliklendirme sonucu ve tüm bileşenlerin tam ablasyon çıktıları mevcut pakette bulunmuyor. Önceki belgelerde bunlara dayanan sayısal iddialar ana sunumdan kaldırıldı. İlgili betikler bu deneyleri üretebilir, betik varlığı deneyin yapılmış olduğunu doğrulamaz.

Arşiv CSV'lerinde seçili test listeleri, yakınsama geçmişi, başlangıç popülasyonu puanı ve tam/seçili süre toplamları yoktur. Çağrı sütunu önceki betiğin planlanan bütçesidir. Bunlar arşivden ölçülmüş gerçek iz gibi gösterilmez.

Sentetik test süreleri toplamı tahmini paket koşum süresidir. Optimizasyonun duvar saatiyle ölçülen arama süresi ayrıca gösterilir. Hata tespit matrisi yoktur. APFD ve gerçek hata keşif başarısı raporlanmaz. IBMGO'nun her ölçütte veya her boyutta GA'dan üstün olduğu iddia edilmez.
