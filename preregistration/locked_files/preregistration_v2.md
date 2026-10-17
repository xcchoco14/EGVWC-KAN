# Pra-registrasi — EGVWC-KAN v2

**Status: DIKUNCI pada 2026-10-08 22:05 WIB**, sebelum unit v2 pertama dijalankan di Kaggle. Disetujui oleh penulis
(tanpa kriteria tambahan jumlah dataset yang dimenangkan). Setelah dikunci, isinya tidak diubah; perubahan apa pun ditulis sebagai *amandemen* bertanggal
di bagian akhir beserta alasannya.

## 1. Latar belakang (hasil v1, 24 dataset × 15 fold, selesai 8 Oktober 2026)
- EGVWC-KAN v1 (baris H): MCC rata-rata 0,502, rank rata-rata 11,5 dari 19 metode. Tidak ada perbedaan signifikan
  setelah koreksi Holm, tetapi v1 tidak melampaui KAN tunggal (0,540), ensemble seragam (0,557), atau ensemble +
  class weight (baris C, 0,566).
- Penyebab yang teridentifikasi dari ablasi: (i) *guard overfitting* (λ × 5) menghasilkan anggota yang datar pada
  dataset kecil; (ii) skor v1 = loss validasi + dispersi logit memberi bobot terbesar kepada anggota datar karena
  dispersinya ≈ 0; (iii) skor dihitung pada split validasi 20% (±20 baris pada dataset terkecil). Proporsi fold
  *collapse* (semua prediksi satu kelas): 1,7% (baris D) → 10,3% (+ guard) → 15,8% (+ bobot EGVW).
- Uji pra-registrasi v1 (skor terstandarisasi vs γ = 1, shard B): tidak terkonfirmasi (Δ = +0,015, p = 0,358).

## 2. Definisi v2 (dibekukan di dokumen ini)
Untuk satu outer fold, dengan konfigurasi KAN (hidden, grid, k, lr, λ, steps) yang dipilih Part 1 pada fold yang sama:
1. **Anggota**: M = 5 KAN, seed `seed·100 + m` (sama dengan v1). Anggota m dilatih pada *bootstrap stratified*
   dari seluruh training fold (setiap kelas di-resample dengan pengembalian sebanyak jumlahnya sendiri). Bila dataset
   memiliki grup (pasien / baris identik), yang di-resample adalah grup, sehingga satu grup tidak pernah berada di
   in-bag dan out-of-bag sekaligus.
2. **Imbalance**: class weight bertingkat dengan eksponen penuh menurut IR (0 / 0,5 / 1 / 1,5; batas 10 × median),
   tanpa *balanced bootstrap* (yang menyisakan hampir nol baris minoritas OOB).
3. **Tanpa** *fitting guard*, tanpa split validasi 80/20, tanpa refit.
4. **Skor anggota** sₘ = *balanced log-loss* pada baris OOB anggota itu sendiri (rata-rata, atas kelas, dari rata-rata
   NLL di dalam kelas).
5. **Gate**: anggota yang gagal dilatih atau memprediksi satu kelas saja pada baris OOB-nya (padahal OOB berisi ≥ 2
   kelas) mendapat bobot 0. Bila semua anggota ter-gate → bobot seragam.
6. **Shrinkage**: γ_eff = γ · n / (n + C), dengan n = median ukuran OOB anggota, γ = 1, C = 100.
7. **Bobot**: wₘ ∝ exp(−γ_eff · (sₘ / min s − 1)) atas anggota yang tidak ter-gate; prediksi = softmax(Σ wₘ zₘ).

## 3. Tahap 1 — pengembangan (24 dataset v1, fold identik)
**Tujuan:** memeriksa apakah diagnosis benar dan memilih pengaturan akhir v2. Hasil Tahap 1 dilaporkan sebagai
**eksploratif**, karena desain v2 berasal dari hasil pada dataset yang sama.

Baris yang dihitung (semua dari lima anggota yang sama): `v2`, `v2 − gate`, `v2 − shrink`, `v2 uniform`,
`v2 uniform + gate`, `v2 [v1 score]`; sensitivitas tanpa training ulang γ ∈ {0,5; 1; 2; 4} × C ∈ {0; 25; 100; 400}.
Pembanding dari Part 1 pada fold yang sama: EGVW-KAN v1 (H), Ens+CW (C), Ens-uniform (B), KAN tunggal (A).

Analisis (dataset dengan 15 fold lengkap untuk semua baris; MCC rata-rata per dataset; Wilcoxon dua sisi):
- **Cek diagnosis:** proporsi fold collapse v2 vs H; korelasi bobot vs collapse.
- **Pembanding utama Tahap 1:** v2 vs H dan v2 vs C.
- **Aturan pembekuan (v2-final):** pengaturan dengan MCC rata-rata tertinggi di antara `v2`, `v2 − gate`,
  `v2 − shrink`, dan 15 varian (γ, C) — bila selisih dengan pengaturan default (γ = 1, C = 100, dengan gate) < 0,005
  MCC, default dipertahankan (lebih sederhana dan sudah ditetapkan sebelum melihat hasil).
- **Lanjut ke Tahap 2** bila v2-final melampaui H dengan Wilcoxon p < 0,05 pada Tahap 1. Bila tidak, hasil Tahap 1
  dilaporkan sebagai temuan dan Tahap 2 diputuskan bersama pembimbing.

## 4. Tahap 2 — konfirmasi (dataset baru) — dirinci sebelum dijalankan
- Dataset: kandidat klinis tabular publik yang **belum pernah dipakai** di v1, diaudit dengan aturan R1–R5 yang sama;
  daftar dan hasil audit dikunci di amandemen dokumen ini sebelum model apa pun dilatih pada dataset tersebut.
- Protokol: repeated nested CV 3 × 5 (inner 3-fold), 12 kandidat, seleksi MCC, fold dan seed dari BASE_SEED.
- Metode: v2-final, EGVW-KAN v1, Ens+CW (C), KAN tunggal, CatBoost, Random forest, Logistic regression, FastKAN —
  anggaran tuning sama.
- **Hipotesis primer:** v2-final memiliki MCC lebih tinggi daripada EGVW-KAN v1 (Wilcoxon dua sisi pada rata-rata
  per dataset, α = 0,05). **Sekunder** (Holm): v2-final vs C, vs KAN tunggal, vs CatBoost; balanced accuracy, AUROC, Brier.
- Semua hasil dilaporkan, termasuk bila hipotesis primer tidak terkonfirmasi.

## 5. Yang tidak boleh dilakukan
- Mengubah definisi v2-final setelah hasil Tahap 2 terlihat.
- Mengeluarkan dataset Tahap 2 setelah audit selain karena aturan R1–R5.
- Menggabungkan hasil Tahap 1 ke dalam uji konfirmasi Tahap 2.

## 6. Penguncian
- Dikunci pada: 2026-10-08 22:05 WIB
- SHA-256 dokumen ini dan kode v2 saat dikunci: lihat `preregistration_v2.sha256` (dibuat bersamaan).

## Amandemen

### Amandemen 1 — 2026-10-09 08:56 WIB — hasil Tahap 1, keputusan lanjut, dan rancangan Tahap 2
Versi dokumen sebelum amandemen ini disimpan utuh sebagai `preregistration_v2_locked_2026-10-08.md`; hash-nya sama
dengan entri `preregistration_v2.md` di `preregistration_v2.sha256` (dikunci 2026-10-08 22:05 WIB).

**Hasil Tahap 1** (`hasil/tahap1_result.json`, 24 dataset × 15 fold, eksploratif): proporsi fold collapse 15,8% (v1)
→ 0,6% (v2). MCC v2 0,565 vs EGVW-KAN v1 0,502 (Δ +0,063, menang/kalah 14/9, Wilcoxon p = 0,070) dan vs Ens+CW 0,566
(Δ −0,001, p = 0,76). Aturan pembekuan: varian terbaik (γ = 0,5, C = 25) hanya +0,002 di atas default → **v2-final =
definisi default** (γ = 1, C = 100, gate aktif), tidak berubah dari Bagian 2.

**Keputusan lanjut ke Tahap 2.** Syarat lanjut (v2-final mengalahkan v1 dengan p < 0,05) **tidak terpenuhi** (p = 0,070).
Sesuai Bagian 3, keputusan diambil oleh penulis: Tahap 2 tetap dijalankan karena arah efek konsisten dan besar pada
dataset yang sebelumnya collapse. Keputusan ini dicatat terbuka dan akan dilaporkan di naskah, termasuk bahwa syarat
lanjut yang dipra-registrasi tidak terpenuhi.

**Dataset Tahap 2 (dikunci di sini, sebelum model apa pun dilatih pada dataset ini).** 13 dataset klinis tabular publik
yang belum pernah dipakai di v1 maupun Tahap 1; semua lolos aturan R1–R5 yang sama (`tahap2_konfirmasi/dataset_inclusion_tahap2.csv`):
Heart Failure Clinical Records, HCV Data, Haberman's Survival, Diabetic Retinopathy Debrecen, Thoracic Surgery,
Breast Cancer (Ljubljana), Fertility, Glioma Grading, Cervical Cancer Risk Factors, Parkinson's (Little 2008),
Chronic Kidney Disease Risk (Bangladesh), AIDS Clinical Trials (ACTG 175), Pediatric Appendicitis (Regensburg).
Koreksi (ditetapkan sebelum training; rinci di `cell3b_datasets_tahap2.py`): kolom turunan outcome atau pasca-baseline
dibuang (Heart Failure: time; ACTG 175: time, cd420, cd820, offtrt, treat; Appendicitis: Length_of_Stay dan target lain;
Cervical: Hinselmann, Schiller, Citology; CKD Bangladesh: stage, affected); label ambigu dibuang (HCV: 7 'suspect donor');
rekaman berulang per pasien dikelompokkan (Parkinson's 2008: per subjek). Kandidat yang tidak dipakai: CKD (id 336, sudah
dipakai di v1) dan Thyroid (id 102, tidak tersedia lewat API).

**Pelaksanaan.** Tiga notebook Kaggle dengan fold identik dan `PIPELINE_VERSION = 2026-10-09-tahap2` (tidak dapat
tergabung dengan hasil v1/Tahap 1): `t2-kan-a`, `t2-kan-b` (EGVW-KAN v1, ladder A–K, dan baris v2-final dalam unit yang
sama, sehingga v2 memakai konfigurasi KAN yang dipilih pada fold yang sama) dan `t2-comp` (8 baseline + 8 varian KAN).
Selain metode yang disebut di Bagian 4, baseline dan varian lain ikut dijalankan; hasilnya dilaporkan deskriptif dan
**tidak** mengubah hipotesis primer/sekunder.

**Analisis Tahap 2 (tidak berubah dari Bagian 4).** Primer: v2-final vs EGVW-KAN v1, MCC rata-rata per dataset, Wilcoxon
dua sisi, α = 0,05, atas dataset dengan 15 fold lengkap. Sekunder (Holm): v2-final vs Ens+CW, vs KAN tunggal, vs CatBoost;
balanced accuracy, AUROC, Brier untuk pasangan primer.

**Hash file Tahap 2 saat amandemen ini dikunci:** lihat `preregistration_v2_amendment1.sha256`.
