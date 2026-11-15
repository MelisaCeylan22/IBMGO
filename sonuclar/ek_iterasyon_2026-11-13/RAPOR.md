# İterasyon deneyi

90/90 IBMGO koşumu tamamlandı.

Toplam iterasyon, katsayı ve bit değişim takvimini de değiştirir. Bu deney farklı bütçe ve kontrol takvimlerini karşılaştırır.

Koddan doğrulama `kod/algoritmalar/bmgo.py` içinde yapılır. `a = -1 - iteration / max_iter` ve `f_vector` içindeki `iteration / max_iter` hareketleri toplam iterasyona bağlar. `progress = iteration / max(1, max_iter - 1)` adaptif bit değişim oranını belirler. `_reference_mgo_coefficients` içindeki `(iteration + 1) / max_iter` de referans hareket kontrolünü değiştirir.

Eşleşen ilk 20 iterasyon eğrisi 0/60 karşılaştırmada aynı kaldı. Bu gözlem tek başına rastgele yolun bütünüyle aynı olduğu anlamına gelmez; kontrol takvimi toplam bütçeye bağlıdır.

250 testte 20→50 geçişinde ortalama puan farkı 0.007011 (20 − yeni). 14 çift iyileşti, 0 eşit kaldı, 1 çift kötüleşti. Ortalama ek arama 2.27 sn, süre çarpanı 2.65.

250 testte 20→100 geçişinde ortalama puan farkı 0.009134 (20 − yeni). 14 çift iyileşti, 0 eşit kaldı, 1 çift kötüleşti. Ortalama ek arama 5.15 sn, süre çarpanı 4.80.

500 testte 20→50 geçişinde ortalama puan farkı 0.005629 (20 − yeni). 13 çift iyileşti, 0 eşit kaldı, 2 çift kötüleşti. Ortalama ek arama 4.98 sn, süre çarpanı 2.32.

500 testte 20→100 geçişinde ortalama puan farkı 0.008682 (20 − yeni). 15 çift iyileşti, 0 eşit kaldı, 0 çift kötüleşti. Ortalama ek arama 13.14 sn, süre çarpanı 4.45.

Bulgular bu sentetik havuzlar, başlangıçlar ve çalışma ortamına aittir. Ortalama farklar tek başına anlamlı üstünlük veya evrensel bir iterasyon sınırı göstermez.

Yakınsama görselleri veri tohumu bazındadır. Her çizgi aynı veride beş başlangıcın ortalaması, bant minimum–maksimum aralığıdır. Eğriler gerçek çağrı numaralarına yerleştirilmiştir.
