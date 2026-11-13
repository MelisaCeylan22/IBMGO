# IBMGO tekrar edilebilirlik deneyleri

Bu çalışma mevcut IBMGO, GA ve Greedy algoritmalarını **yeniden ayarlamadan**
karşılaştırır. Veriler gerçek operatör verisi değil, oluşturulmuş telekom test
senaryolarıdır.

## Ne yaptık?

- Paket boyutları: 50, 100, 250, 500 ve 1000 test.
- Üç farklı veri kümesi başlangıcı: 83, 97 ve 109.
- Her veri kümesinde IBMGO ve GA için optimizer başlangıçları 42–51
  (10 tekrar). Greedy deterministik olduğu için veri kümesi başına bir kez.
- IBMGO ve GA'nın koşum başına fitness aday değerlendirme bütçesi 1212.
- Bütün sonuçlar: kritik kapsam %100, toplam kapsam en az %90,
  ön koşul ihlali sıfır koşulundan geçmek zorundadır.

## Windows'ta nasıl yeniden çalıştırılır?

Proje klasöründeki PowerShell terminalinde:

```powershell
pip install numpy matplotlib
python -B -m kod.deneyler.ablation --sizes 50 100 250 500 1000 --runs 10 --data-seed 83 --output-dir sonuclar/dayaniklilik_83 --variants BMGO_boyuta_adaptif
python -B -m kod.deneyler.ablation --sizes 50 100 250 500 1000 --runs 10 --data-seed 97 --output-dir sonuclar/dayaniklilik_97 --variants BMGO_boyuta_adaptif
python -B -m kod.deneyler.ablation --sizes 50 100 250 500 1000 --runs 10 --data-seed 109 --output-dir sonuclar/dayaniklilik_109 --variants BMGO_boyuta_adaptif
python -B -m kod.deneyler.robustness_analysis --inputs sonuclar/dayaniklilik_83/ablasyon_veri83.csv sonuclar/dayaniklilik_97/ablasyon_veri97.csv sonuclar/dayaniklilik_109/ablasyon_veri109.csv
```

Ardından `sonuclar/dayaniklilik_analizi` klasöründeki Markdown raporu, CSV'ler,
PNG grafikler ve hazırlanmış Excel dosyasını açın. Grafikler ve rapor ilk üç
komutun CSV'leri üzerinden otomatik üretilir.

## Sunumda öne çıkarılacak sınırlar

- Fitness değeri **düşükse** daha iyi, süre tasarrufu **yüksekse** daha iyi.
- Ortalamayı ve standart sapmayı birlikte gösterin; tek başarılı koşumu seçmeyin.
- Greedy çalışma hızı düşük olabilir, fakat fitness değerlendirme bütçesi
  IBMGO / GA ile aynı değildir; başarı kalitesiyle hesaplama maliyetini ayırın.
- Gerçek hata tespit matrisi olmadığı için **APFD hesaplanmamıştır**.
- Yalnız üç yeni sentetik veri kümesi kullanılmıştır; saha başarısı ispatı değildir.
