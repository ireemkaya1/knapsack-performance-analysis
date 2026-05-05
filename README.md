# Knapsack (Sırt Çantası Problemi) Performans Analizi

Bu proje, Algoritma Analizi ve Tasarımı dersi kapsamında hazırlanmıştır. Çalışmada 0/1 Knapsack problemi için kesin çözüm üreten Dinamik Programlama (DP) yöntemi ile yaklaşık/sezgisel çözüm üreten Simulated Annealing (SA), Greedy ve Greedy-SA yöntemleri karşılaştırılmıştır.

Ayrıca büyük ölçekli veri setinde kesin referans değer elde etmek için Branch and Bound (B&B) yöntemi kullanılmıştır.

## Kullanılan Algoritmalar

### 1. Dynamic Programming (DP)

DP yöntemi küçük ve orta ölçekli veri setlerinde optimum referans çözüm üretmek için kullanılmıştır. Bu çalışmada N=100 ve N=1000 veri setlerinde referans değer DP ile alınmıştır.

Klasik 2D tablo tabanlı DP yaklaşımının zaman ve bellek maliyeti problem boyutu ve kapasite değeriyle birlikte artar. Bu nedenle büyük ölçekli veri setinde DP ayrıca ölçeklenebilirlik açısından incelenmiştir.

### 2. Simulated Annealing (SA)

SA, yaklaşık çözüm üreten meta-sezgisel bir yöntemdir. Komşu çözümler üzerinden arama yaparak kabul olasılığı ve sıcaklık parametresi yardımıyla daha iyi çözümlere ulaşmaya çalışır.

### 3. Greedy

Greedy yöntemi, nesneleri değer/ağırlık oranına göre sıralayarak hızlı bir başlangıç çözümü üretir. Çok hızlıdır fakat her zaman optimum sonucu garanti etmez.

### 4. Greedy-SA

Greedy-SA yöntemi, Greedy ile bulunan başlangıç çözümünü SA ile iyileştirmeyi amaçlar. Böylece hızlı başlangıç çözümü ile meta-sezgisel arama birlikte kullanılmış olur.

### 5. Branch and Bound (B&B)

Branch and Bound yöntemi, özellikle büyük veri setinde kesin referans değer elde etmek için kullanılmıştır. Bu projede N=10000 veri seti için referans değer B&B ile alınmıştır.

## Veri Setleri

Deneylerde üç farklı problem boyutu kullanılmıştır:

| Veri Seti | Eleman Sayısı |
|---|---:|
| Küçük | N = 100 |
| Orta | N = 1000 |
| Büyük | N = 10000 |

## Deneysel Değerlendirme

Algoritmalar aynı veri setleri üzerinde çalıştırılmış ve şu ölçütlerle karşılaştırılmıştır:

- Bulunan toplam değer
- Toplam ağırlık
- Çalışma süresi
- Referans değere göre doğruluk açığı

Doğruluk açığı şu formülle hesaplanmıştır:

```text
Gap (%) = ((Referans Değer - Algoritma Değeri) / Referans Değer) × 100
