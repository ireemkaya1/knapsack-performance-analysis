# Knapsack (Sırt Çantası Problemi) Performans Analizi

Bu proje, Algoritma Analizi ve Tasarımı dersi kapsamında hazırlanmıştır. Çalışmada **Knapsack (Sırt Çantası Problemi)** için kesin çözüm üreten **Dinamik Programlama (DP)** yöntemi ile yaklaşık/sezgisel çözüm üreten **Greedy**, **Simulated Annealing (SA)** ve **Greedy-SA** yöntemleri karşılaştırılmıştır.

Amaç, problem boyutu büyüdükçe kesin çözüm veren DP yönteminin ne zaman pratik olmaktan çıktığını ve sezgisel yöntemlerin ne kadar hızlı çözüm ürettiğini incelemektir.

---

## Kullanılan Algoritmalar

### 1. Dynamic Programming (DP)

DP yöntemi kesin çözüm üretir. Küçük ve orta ölçekli veri setlerinde optimum sonuç referansı olarak kullanılmıştır. Ancak problem boyutu büyüdükçe zaman ve bellek maliyeti arttığı için büyük veri setlerinde pratik olmaktan çıkmaktadır.

### 2. Greedy

Greedy yöntemi nesneleri değer/ağırlık oranına göre sıralar ve çanta kapasitesi dolana kadar seçim yapar. Çok hızlıdır fakat her zaman optimum sonucu garanti etmez.

### 3. Simulated Annealing (SA)

SA, yaklaşık çözüm arayan meta-sezgisel bir yöntemdir. Rastgele komşu çözümler üzerinden arama yaparak daha iyi çözüme ulaşmaya çalışır. Büyük veri setlerinde DP’ye göre çok daha kısa sürede sonuç verir.

### 4. Greedy-SA

Greedy-SA yöntemi, Simulated Annealing algoritmasını greedy çözümden başlatır. Böylece başlangıç çözümü daha güçlü olur ve çözüm kalitesi genellikle Greedy yönteminden kötüye gitmez.

---

## Veri Setleri

Deneylerde üç farklı problem boyutu kullanılmıştır:

| Veri Seti | Eleman Sayısı |
|---|---:|
| Küçük | N = 100 |
| Orta | N = 1000 |
| Büyük | N = 10000 |

Aynı veri setleri üzerinde DP, Greedy, SA ve Greedy-SA algoritmaları çalıştırılmıştır.

---

## Deneysel Bulgular

- **N = 100** için DP başarıyla çalışmış ve optimum çözüm üretmiştir.
- **N = 1000** için DP yine çalışmış ve optimum referans değer elde edilmiştir.
- **N = 10000** için DP 1800 saniyelik zaman sınırı içinde tamamlanamamış ve timeout durumuna düşmüştür.
- Greedy, SA ve Greedy-SA yöntemleri büyük veri setinde de kısa sürede sonuç üretmiştir.
- Doğruluk açığı (accuracy gap), DP’nin başarıyla çalıştığı veri setlerinde DP sonucu referans alınarak hesaplanmıştır.
- N = 10000 için DP optimum değeri bulunamadığından doğruluk açığı N/A olarak değerlendirilmiştir.

Bu sonuçlar, veri boyutu büyüdükçe kesin çözüm veren DP yönteminin pratik maliyetinin arttığını; sezgisel yöntemlerin ise optimum garantisi vermeden daha hızlı ve uygulanabilir çözümler ürettiğini göstermektedir.

---

## Proje Yapısı

```text
knapsack-performance-analysis/
│
├── code/
│   ├── algorithms.py
│   ├── config.py
│   ├── experiment.py
│   ├── main.py
│   ├── plot.py
│   └── validate_outputs.py
│
├── results/
│   ├── experiment_results.csv
│   ├── validation_report.txt
│   ├── plots/
│   └── solutions/
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── requirements.txt
└── README.md
