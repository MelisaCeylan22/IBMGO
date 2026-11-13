# IBMGO sonuçları ve sunum yorumu

Ana deneyin doğrulanabilir kaynağı `sonuclar/dayaniklilik_83/ablasyon_veri83.csv`, `dayaniklilik_97/ablasyon_veri97.csv`, `dayaniklilik_109/ablasyon_veri109.csv` dosyalarıdır. Özet ve eşleşmiş karşılaştırmalar `sonuclar/dayaniklilik_analizi` içindedir.

## Deney tasarımı

50, 100, 250, 500, 1000 test; veri tohumları 83, 97, 109; IBMGO/GA tohumları 42–51. Toplam 315 kayıt, 150 IBMGO, 150 GA, 15 Greedy. Greedy veri ve boyut başına bir kez çalışır. Kayıtlarda kritik eksik ve ön koşul ihlali sıfır, genel kapsam en az %90'dır.

Ana amaç ağırlıkları süre 0,40, test sayısı 0,40, tekrar 0,20 ve nadir kayıp 0. Kaynak deney betiği `kod/deneyler/ablation.py` bu ayarları tanımlar. Önceki CSV'ler ayrı girdi snapshot'ı veya kaynak kod sürüm kaydı içermediğinden bütün eski koşumları bit düzeyinde yeniden üretme garantisi verilmez. Yeni sunum motoru bunları ayrı dosyalara kaydeder.

## Bulgular

Düşük fitness daha iyidir. Ana eşleşmiş karşılaştırmada 50 testte 30 eşitlik, 250 testte IBMGO için 24/30, 500 testte 23/30 kazanım bulunur. 1000 testte IBMGO 12, GA 18 eşleşmede daha düşük fitness verir. IBMGO her boyutta üstün değildir. Tam sayısal tablo notebook'ta gerçek CSV'lerden oluşturulur.

Özetlerde ± örnek standart sapması kullanılır. IBMGO/GA boyut başına 30 veri/algoritma kaydından, Greedy 3 veri kaydından hesaplanır. Birleştirilmiş standart sapma veri ve arama rastgeleliğini birlikte içerir. Yeni yakınsama grafikleri önce aynı veri ve boyut içindeki tekrarları gösterir.

Arama süresi ve seçili paketin tahmini koşum süresi farklı ölçütlerdir. Sentetik süreler gerçek CI/CD ölçümü değildir. Fitness değerleri farklı boyutlar arasında doğrudan başarı sıralaması olarak kullanılmaz.

## Kaynak ayrımı

- **Kayıtlı deney** ana arşivdir. İz ve seçili paket bulunmadığında yeni değer uydurulmaz.
- **Yeni çalıştırma** canlı demo veya tam planla gerçekten yeniden hesaplanır. Her çalışma kendi klasörünü, ayarlarını ve girdi kopyalarını taşır.
- **Ek ölçek deneyi** önceki notebook'un 500–1000–5000, beş veri tohumu ve tek algoritma tohumu akışıdır. `IBMGO_Ek_Olcek_Deneyi.ipynb` içinde korunur. Ana dayanıklılık ortalamasına eklenmez.

## Mevcut pakette bulunmayan kanıtlar

Önceki belgelerde seed 71 doğrulamasına, `nihai_bmgo` klasörüne ve seed 42 önceliklendirme sonuçlarına yapılan sayısal göndermeler bu paketteki dosyalarla doğrulanamıyor. Bu nedenle o sayılar doğrulanmış sonuç olarak kullanılmıyor. Tam ilk BMGO ve bileşen ablasyonu da mevcut CSV'lerde yoktur. Dosya adında ablasyon geçmesi tam ablasyon yapıldığını kanıtlamaz.

Seçilen testlerin bağımlılıklara uygun sıralanması yeni koşumda gösterilir. Bu gereksinim kapsamının önceliklendirilmesidir. Hata tespit matrisi bulunmadığından APFD veya gerçek hata yakalama başarısı iddiası yoktur.

Yeni düzenlemenin kontrolleri, gerçekten çalıştırılan küçük deneyler ve bilinen sınırlamalar `SUNUM_DOGRULAMA.md` dosyasında kayıtlıdır.
