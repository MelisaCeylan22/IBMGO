# Ana dayanıklılık deneyini gösterme ve yeniden üretme

Ana arşiv 50, 100, 250, 500, 1000 test; 83, 97, 109 veri tohumları; IBMGO ve GA için 42–51 algoritma tohumlarını içerir. Greedy her boyut ve veri için bir kez çalışır. Toplam 315 kayıt vardır.

## Kayıtlı sonuçlar

`IBMGO_Sunum.ipynb` dosyasını açıp hücreleri sırayla çalıştırın. **Kayıtlı ana deneyi göster** düğmesi üç ham CSV'yi okur ve tasarımı doğrular. Grafikler bu kayıtlardan çizilir. `sonuclar/dayaniklilik_analizi` eski analizin rapor ve grafiklerini korur.

Ham kaynaklar `sonuclar/dayaniklilik_<83|97|109>/ablasyon_veri<tohum>.csv` dosyalarıdır. Dosyalar yalnız IBMGO, GA ve Greedy karşılaştırmasını içerir. Tam bileşen ablasyonu içermez.

## Yeni tam deney

Notebook ayar panelinden **Tam dayanıklılık** seçip **Deneyi çalıştır / Sonuçları güncelle** düğmesine basın. Bu seçim tek başına hesaplama başlatmaz. Varsayılan hızlı demo 50 test / veri 83 / algoritma 42 ile üç koşumdur.

Widgetsız alternatif aynı notebook'un sonundadır. `AYARLAR = Settings.full()` seçin, son hücrede `CALISTIR = True` yapın. Her çalışma ayrı `sonuclar/sunum_calistirmalari` klasörüne ayarları, giriş dizilerini, ham sonuçları ve gerçek değerlendirme izlerini yazar. Önceki arşiv klasörlerini çıktı hedefi olarak kullanmayın.

IBMGO standart planı 1212, GA planı 1212 amaç çağrısıdır. Gerçek çağrılar yeni izlerde ayrıca tutulur. Eski CSV'lerin aday bütçesi sütunu ölçülmüş iz değildir. Eşit değerlendirme sayısı eşit arama süresi anlamına gelmez.

## Sınırlar

Bütün sonuçlar kritik kapsam %100, genel kapsam en az %90 ve sıfır ön koşul ihlaliyle denetlenir. Fitness düşük, süre tasarrufu yüksek olduğunda daha iyidir. Ortalama ile örnek standart sapmasını birlikte okuyun. Greedy'nin 3, popülasyon yöntemlerinin boyut başına 30 kaydı vardır.

Veriler ve test süreleri sentetiktir. Hata tespit matrisi bulunmadığından APFD hesaplanmaz. Üç veri kümesi saha başarısının kanıtı değildir. Ayrıntılı kaynak kontrolü ve küçük doğrulama deneyleri `SUNUM_DOGRULAMA.md` içindedir.
