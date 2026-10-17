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
_(belum ada)_
