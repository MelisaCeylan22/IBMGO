# Sunum düzenlemesi doğrulama kaydı

13 Kasım 2026 tarihinde yerel proje üzerinde düzenleme ve kontroller yapıldı.

## Kaynaklardan doğrulanan durum

- Ana arşiv 50, 100, 250, 500, 1000 boyutlarını ve 83, 97, 109 veri tohumlarını içeriyor. Her veri ve boyutta IBMGO/GA için 42–51 algoritma tohumları, Greedy için bir kayıt var. Toplam 315 kayıt doğrulandı.
- Ham dosyalar `ablasyon_veri<tohum>.csv` adını taşıyor. Ayrı `algoritma_karsilastirma.csv` dosyaları bu klasörlerde yok. Bu bir kaynak adı uyuşmazlığıdır.
- Bu CSV'ler yalnız `BMGO_boyuta_adaptif` (sunum adı IBMGO), GA ve Greedy içeriyor. Tam bileşen ablasyonu içermez.
- Ham kayıtlardan hesaplanan 15 yöntem/boyut özetinin ortalama ve örnek standart sapmaları, eski `algoritma_ozet.csv` ile CSV yuvarlaması düzeyinde uyuştu. Fitness, tasarruf, seçilen test, genel kapsam ve arama süresi denetlendi.
- Eski notebook farklı bir planı, 500–1000–5000 test / beş veri tohumu / tek algoritma tohumu / A-B havuzlarını çalıştırıyordu. Bu kaynak ayrı `IBMGO_Ek_Olcek_Deneyi.ipynb` dosyasına taşındı. Kod hücreleri korundu, eski gösterim çıktıları temizlendi. Büyük ek deney yeniden çalıştırılmadı.
- Seed 71, `nihai_bmgo` ve seed 42 önceliklendirme sonuçları mevcut pakette yok. Belgelerdeki doğrulanamayan sayısal iddialar kaldırıldı ve eksiklikler açıkça yazıldı.

## Yapılan hesaplamalar

| Yeni kayıt klasörü | Boyut | Veri | Algoritma tohumları | Tamamlanan koşum |
|---|---|---|---|---|
| `20261113T121057Z_c1bc271d` | 50 | 83 | 42 | 3 |
| `20261113T121357Z_437609a1` | 250 | 83 | 42, 43 | 5 |

Klasörler `sonuclar/sunum_calistirmalari` altındadır. Ayar JSON'u, giriş NPZ'leri, veri açıklamaları, ham/özet CSV'leri ve çağrı/seçim izleri saklandı. Sonuç ve yakınsama PNG'leri de bu küçük çalışmalardan üretildi, görsel olarak incelendi. Notebook bu PNG'leri yeni sonuç yerine kullanmaz, grafikleri tablodan ve izden çizer.

50 testlik çalışmada IBMGO ve GA başlangıçta nihai puana erişti. Bunun sonucu düz bir eğridir. 250 testlik çalışmada iki yöntemin gerçek iyileşme eğrileri kaydedildi. Seçilen koşumun farklı tohumlardaki eğrileri aynı veri/boyut içinde silik gösterilir.

Tam 315 koşumluk dayanıklılık deneyi yeniden çalıştırılmadı. 1000 ve 5000 testlik yeni arama yapılmadı. Ana dayanıklılık tabloları mevcut arşivden okunur. Sekiz yeni kalıcı koşum arşiv ortalamalarına eklenmedi.

## Otomatik ve uçtan uca kontroller

`python -B -m unittest discover -s kod/testler -q` ile 21 test geçti. Bunların 13'ü mevcut, 8'i sunum düzenlemesinin regresyon kontrolleridir.

- Mevcut üretici ve arama politikası testleri geçti.
- İz açık/kapalı aynı seed ile aynı çözümü ve eğriyi verdi. Eski üçlü dönüş sözleşmesi korundu.
- Amaç fonksiyonu çağrıları bağımsız sayımla karşılaştırıldı. Başlangıç, yerel ve spiral adaylar sayıldı. Her adımın son iz değeri döndürülen eğriyle uyuştu.
- İlk erişim ve son yerel iyileşmenin toleransla hesaplanması kontrol edildi.
- Aynı düzenlenmiş süre, kapsam ve kritik hedef dizilerinin üç yönteme ulaştığı kontrol edildi. Sonuç kayıtlarında ortak girdi hash'i doğrulandı.
- Geçersiz süre, kapsam ve ön koşullar aramadan önce reddedildi.
- Sonuç grafiğinin çubuk değerleri ham tablodan, yakınsama başlangıcı izden doğrulandı.
- Ayar veya veri kopyası değişince eski sonuç güncel sayılmıyor. Filtreler yeni optimizasyon başlatmıyor.
- Ana arşiv dosyaları ve yedeklerde Git farkı olmadığı doğrulandı.

Ana notebook'un 7 kod hücresi Python çekirdeğinde hatasız çalıştırıldı ve kayıtlı ana sonuçlarla kaydedildi. Açılışta yeni deney başlamadığı doğrulandı.

Ayrı, geçici bir notebook çekirdeğinde gerçek widget düğmeleriyle test süresi değiştirildi, uygulandı, üç yöntem çalıştırıldı, bütün sonuç grafikleri ve yakınsama çizildi. Filtrelerin yeni klasör/hesaplama üretmediği, ağırlık değişiminin güncelliği bozduğu ve geçersiz JSON ayarının düğmeyi kapattığı doğrulandı. Geçici çıktı klasörleri otomatik temizlendi. Bu kontrol sırasında ilk menü seçiminin boş kalması hatası bulundu, düzeltildi ve akış tekrar başarıyla çalıştırıldı.

## Değişen dosyalar

- `IBMGO_Sunum.ipynb` ana akış, arşiv görünümü ve ortak panel.
- `IBMGO_Ek_Olcek_Deneyi.ipynb` ayrılan önceki deney.
- `kod/algoritmalar/bmgo.py`, `baselines.py` isteğe bağlı gerçek çağrı izleme.
- `kod/algoritmalar/izleme.py` başlangıç, ilk erişim ve son iyileşme kaydı.
- `kod/deneyler/sunum_deneyi.py` ortak ayarlar, doğrulama, çalışma/kayıt motoru ve grafikleri.
- `kod/raporlama/sunum_paneli.py` ayar, veri düzenleme ve sonuç ekranları.
- `kod/testler/test_sunum.py` regresyon kontrolleri.
- `PROJE_KILAVUZU.md`, `SONUCLAR_VE_YORUM.md`, `DAYANIKLILIK_DENEY_KILAVUZU.md` tutarlı kaynak ve kullanım açıklamaları.
- Bu doğrulama kaydı ve iki yeni küçük deney klasörü.

Bu düzenleme yereldir. GitHub'a gönderme veya commit yapılmadı.


## Sonuç ekranının sadeleştirilmesi · güncel revizyon

13 Kasım 2026 tarihinde sonuç ekranı yeniden düzenlendi.

- Ana arşiv ve hesaplama kaynaklarına ait 28 dosyanın SHA-256 özetleri düzenleme öncesiyle aynı. Ana plan 5 boyut, 3 veri tohumu ve IBMGO/GA için 10 tekrar olarak korundu. Arşiv 315 kayıt içeriyor.
- Genel özet ve tek havuz ayrıntısı ayrı sekmelerde. Ayrıntıda deney özeti, yöntem tablosu, önce–sonra grafiği ve seçilebilir iterasyon görünümü var. Ayarlar, kaynak yolları, ham kayıtlar, değerlendirme olayları ve seçili testler kapalı teknik bölümlerde.
- Karşılaştırma tablosu gerçek satırlardan ortalama ve örnek standart sapma hesaplıyor. Vurgu ölçüt bazında ve gösterim hassasiyetindeki eşitlikleri koruyor. Başarı etiketi üç kısıt kontrolüne bağlı.
- Arşivde mutlak süre ve iz bulunmadığı için süre oranı gösteriliyor. İterasyon bölümü bir açıklama içeriyor. Eksik süre yerine sıfır üretilmiyor.
- Varsayılan yeni demo 250 test, veri tohumu 83 ve algoritma tohumu 42. Standart popülasyon, iterasyon, arama politikası ve amaç fonksiyonu korunuyor.
- Gerçek panel düğmesiyle üç yöntem çalıştırıldı. İlk çalışmada IBMGO'nun 20 iterasyonluk izinde nihai puana ilk erişim 12. iterasyon. İşaret, başlangıç dahil gerçek eğriyle ve değerlendirme iziyle karşılaştırıldı.
- Süre verisi düzenlenip tekrar üç yöntem çalıştırıldı. Sonuç kimlikleri, tablolar ve grafikler yeni sonuç nesnesinden güncellendi. Geçersiz süre arama başlamadan reddedildi ve son geçerli sonuç korundu.
- Yöntem, veri, büyüklük ve iz görünümü filtreleri izlenerek yeni optimizasyon başlatmadıkları doğrulandı. Eski widgetlar değiştirilip kapatılıyor. Sonuçlar alta eklenmiyor.
- Tek koşum ve çoklu tekrar etiketleri, küçük arama süreleri, yuvarlama eşitlikleri, başlangıçta düz iz ve eksik arşiv bilgisi için kontroller eklendi. Toplam 28 test geçti.
- Notebook temiz Python çekirdeğinde çalıştırılıp ana arşiv görünümü ve gerçek PNG widgetlarıyla kaydedildi. Genel özet, canlı ayrıntı, arşiv ayrıntısı ve eşit değerlendirme görünümü tarayıcıda görüntülenerek kontrol edildi.

Bu revizyondaki gerçek çalışmalar ayrı kayıt klasörlerinde bulunur.

- `sonuclar/sunum_calistirmalari/20261113T132652Z_a5268357` orijinal 250/83/42 çalışması.
- `sonuclar/sunum_calistirmalari/20261113T132703Z_c72003bf` süre düzenlemesinden sonraki çalışma.

Sunum sırası genel özet, 250 testin arşiv ayrıntısı ve yeni 250/83/42 çalışmasının iki iz görünümüdür.
