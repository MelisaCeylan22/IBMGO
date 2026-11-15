# Genişletilmiş iterasyon deneyi

90 eski ve 90 yeni IBMGO koşumu uyum kontrolünden sonra 180 koşum olarak özetlendi. Yeni koşumlar 1000 ve 5000 testte gerçekten yeniden çalıştırıldı.

Toplam iterasyon, katsayı ve bit değişim takvimini de değiştirir. Bu deney farklı bütçe ve kontrol takvimlerini karşılaştırır.

Hesaplama kaynaklarının içerik özetleri ve kütüphane sürümleri aynı. Ağırlıklar, onarım, veri üretimi ve boyuta duyarlı politika korunmuştur. Yeni çalıştırıcı yalnızca boyut planını ve kayıt akışını genişletir.

Tahmini süre tasarrufu sentetik test sürelerinden hesaplanır. Arama süresi bu bilgisayarda, gerçek iz kaydı açıkken ölçülmüştür; gerçek bir CI/CD hızlanması ölçümü değildir.

## 250 test

Ortalama amaç puanı / arama süresi: 20 iterasyonda 0.300283 / 1.37 sn; 50 iterasyonda 0.293272 / 3.64 sn; 100 iterasyonda 0.291149 / 6.52 sn.

20→50 geçişinde ortalama fitness farkı 0.007011 (önceki − yeni), iyileşen/eşit/kötüleşen 14/0/1. Tasarruf farkı +0.331 yüzde puan. Ek arama 2.27 sn, eşleşmiş süre çarpanı ortalaması 2.65.

20→100 geçişinde ortalama fitness farkı 0.009134 (önceki − yeni), iyileşen/eşit/kötüleşen 14/0/1. Tasarruf farkı +0.507 yüzde puan. Ek arama 5.15 sn, eşleşmiş süre çarpanı ortalaması 4.80.

50→100 geçişinde ortalama fitness farkı 0.002123 (önceki − yeni), iyileşen/eşit/kötüleşen 11/0/4. Tasarruf farkı +0.176 yüzde puan. Ek arama 2.88 sn, eşleşmiş süre çarpanı ortalaması 1.81.

## 500 test

Ortalama amaç puanı / arama süresi: 20 iterasyonda 0.255055 / 3.93 sn; 50 iterasyonda 0.249426 / 8.91 sn; 100 iterasyonda 0.246372 / 17.06 sn.

20→50 geçişinde ortalama fitness farkı 0.005629 (önceki − yeni), iyileşen/eşit/kötüleşen 13/0/2. Tasarruf farkı +0.398 yüzde puan. Ek arama 4.98 sn, eşleşmiş süre çarpanı ortalaması 2.32.

20→100 geçişinde ortalama fitness farkı 0.008682 (önceki − yeni), iyileşen/eşit/kötüleşen 15/0/0. Tasarruf farkı +0.542 yüzde puan. Ek arama 13.14 sn, eşleşmiş süre çarpanı ortalaması 4.45.

50→100 geçişinde ortalama fitness farkı 0.003053 (önceki − yeni), iyileşen/eşit/kötüleşen 11/0/4. Tasarruf farkı +0.144 yüzde puan. Ek arama 8.15 sn, eşleşmiş süre çarpanı ortalaması 1.93.

## 1000 test

Ortalama amaç puanı / arama süresi: 20 iterasyonda 0.219042 / 7.92 sn; 50 iterasyonda 0.213794 / 18.73 sn; 100 iterasyonda 0.211794 / 36.79 sn.

20→50 geçişinde ortalama fitness farkı 0.005248 (önceki − yeni), iyileşen/eşit/kötüleşen 13/0/2. Tasarruf farkı +0.368 yüzde puan. Ek arama 10.81 sn, eşleşmiş süre çarpanı ortalaması 2.37.

20→100 geçişinde ortalama fitness farkı 0.007248 (önceki − yeni), iyileşen/eşit/kötüleşen 14/0/1. Tasarruf farkı +0.479 yüzde puan. Ek arama 28.88 sn, eşleşmiş süre çarpanı ortalaması 4.65.

50→100 geçişinde ortalama fitness farkı 0.002000 (önceki − yeni), iyileşen/eşit/kötüleşen 12/0/3. Tasarruf farkı +0.110 yüzde puan. Ek arama 18.07 sn, eşleşmiş süre çarpanı ortalaması 1.97.

## 5000 test

Ortalama amaç puanı / arama süresi: 20 iterasyonda 0.143944 / 28.94 sn; 50 iterasyonda 0.140558 / 68.05 sn; 100 iterasyonda 0.139832 / 132.44 sn.

20→50 geçişinde ortalama fitness farkı 0.003386 (önceki − yeni), iyileşen/eşit/kötüleşen 14/0/1. Tasarruf farkı +0.088 yüzde puan. Ek arama 39.11 sn, eşleşmiş süre çarpanı ortalaması 2.39.

20→100 geçişinde ortalama fitness farkı 0.004111 (önceki − yeni), iyileşen/eşit/kötüleşen 15/0/0. Tasarruf farkı +0.118 yüzde puan. Ek arama 103.49 sn, eşleşmiş süre çarpanı ortalaması 4.64.

50→100 geçişinde ortalama fitness farkı 0.000726 (önceki − yeni), iyileşen/eşit/kötüleşen 9/0/6. Tasarruf farkı +0.030 yüzde puan. Ek arama 64.39 sn, eşleşmiş süre çarpanı ortalaması 1.95.

Bu kayıtlarda 20 iterasyon en kısa aramayı sağlar. 50 iterasyon kalite–süre dengesi için değerlendirilebilir. 100 iterasyonun daha düşük ortalama puanı ek süre karşılığında elde edilir; tek tek her koşumun iyileşmesi garanti değildir.

Kısa arama ile daha yüksek bütçeli arama arasındaki seçim, bu kayıtlardaki ek kalite ve ek hesaplama süresi birlikte değerlendirilerek yapılır. Bulgular bu havuz ve başlangıçlarla sınırlıdır.

Önceki 1212 değerlendirmeli GA kayıtları daha yüksek bütçeli IBMGO ile üstünlük kıyası için kullanılmadı. Ölçek deneyi kendi bölümünde ve kendi kayıtlarında korunur.

CSV dosyaları ham kayıtları, 12 satırlık iterasyon özetini ve 180 eşleşmiş geçişi içerir. Min–max bantları güven aralığı değildir; eğriler gerçek çağrılarda biter.
