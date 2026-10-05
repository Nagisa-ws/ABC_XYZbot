# Laporan Audit & Perubahan — curriculum_arm_rl

> **Dokumen historis.** Ini adalah audit kode awal, ditulis sebelum training penuh Misi 1-3 dijalankan.
> Bagian "Belum tervalidasi" di bawah sudah **usang**: training penuh telah dijalankan dan hasilnya,
> termasuk controller hibrida, ada di [README.md](README.md) dan [KONTROL_HIBRIDA.md](KONTROL_HIBRIDA.md).


Dokumen ini merangkum bug yang ditemukan pada kode asli, perbaikan yang dilakukan, hasil uji,
dan hal yang **belum** tervalidasi.

## 1. Bug pada kode asli

### Kompatibilitas antar-skrip (penyebab utama kegagalan curriculum)
| # | Masalah | Dampak |
|---|---|---|
| 1 | Dimensi observasi berbeda per misi (26 / 31 / 36) | `PPO.load` Misi 1→2→3 gagal; transfer bobot mustahil |
| 2 | Skrip eval membangun observasi dengan dimensi/urutan berbeda dari env | Policy menerima input yang tak dikenalnya saat evaluasi |
| 3 | Eval memakai `scene.xml` (model training); XML eval tidak terpakai dan berisi teks sampah | Hasil eval tidak mewakili model eval |
| 4 | Collision gripper di model eval memakai mesh, di training memakai primitif | Perilaku kontak berbeda training vs eval |
| 5 | Mission3 menggandakan fitur observasi; `eval_mission3` meng-hardcode 0.15 / 0.04 | Mismatch env–eval |
| 6 | `shared/__init.py` (salah nama); `leaf_cone_sampler` tak terpakai; statistik VecNormalize tak ditransfer | Paket tidak terdeteksi, kode mati |

### Kebenaran fisika / algoritma
| # | Masalah | Dampak |
|---|---|---|
| 7 | IK `pos_tol` 5 mm = threshold akhir; `mj_forward` penuh di tiap iterasi IK | Gerakan <5 mm tidak dieksekusi; IK lambat |
| 8 | Tidak ada kompensasi gravitasi | Lengan melorot ±10 mm → threshold 5 mm tak tercapai |
| 9 | Kapsul bahu menembus lantai & base di pose home | 2 kontak permanen sejak langkah 0 |
| 10 | Tidak ada terminasi self-collision / lantai | Policy bisa "menang" lewat konfigurasi tak fisis |
| 11 | Cek rintangan hanya memakai origin 4 link | Lengan atas/bawah menembus rintangan tanpa terdeteksi |
| 12 | T2 dan rintangan berasal dari konstelasi berbeda | Skenario Misi 3 tak konsisten |

### Reward & curriculum
| # | Masalah | Dampak |
|---|---|---|
| 13 | Success reward 2000–3000 vs suku lain ~0.1 | Gradien didominasi satu peristiwa langka |
| 14 | Progress asimetris (+50 vs −200, tanda keliru) | Bias perilaku |
| 15 | Smoothness membandingkan target sendi (rad) dengan aksi [-1,1]; Phase B Misi 3 melewatinya | Suku tak bermakna |
| 16 | Threshold T1 (fase transisi) memakai ulang threshold sukses | Transisi fase ikut mengetat saat curriculum naik |
| 17 | Callback adaptive: window reset bisa naik beberapa level sekaligus; config ditulis via file temp | Curriculum tak stabil, rawan race |
| 18 | Tidak ada best-model / eval callback; struktur checkpoint tidak konsisten | Tak ada model terbaik yang layak diekspor |

## 2. Bug yang ditemukan dan diperbaiki saat penulisan ulang
Ditemukan lewat pengujian, bukan dari kode asli:
- **Derau sensor pada dimensi nonaktif**: setelah VecNormalize, dimensi konstan berderau menjadi N(0,1) acak masuk ke policy. Sekarang derau hanya pada dimensi aktif misi itu (transfer kini persis: selisih aksi 0.0).
- **Validasi Misi 3 hanya meloloskan ~8% skenario** (reset 157 ms): badan robot lebih tebal daripada clearance garis TCP. Diganti *reject-or-repair* (lintasan referensi dijalankan sekali; rintangan yang dilanggar dipangkas) → lolos 99%, reset ~10 ms.
- **Bias distribusi target**: 22% titik workspace (seluruhnya di sisi y<0) ditolak karena garis lurus home→T1 melewati base. Ditambah jalur polar mengitari base; distribusi azimuth kini merata.
- **Fallback viewer tidak berfungsi**: kegagalan init GLFW menghentikan proses di level C. Sekarang `DISPLAY` diperiksa dulu.
- **Statistik cross-track = 0 palsu** saat robot belum masuk fase B (menyesatkan di log dan callback) → kini hanya ada bila fase B tercapai.

## 3. Hasil uji (sandbox: 1 CPU, MuJoCo 3.14, SB3 2.9)

| Uji | Hasil |
|---|---|
| Jacobian IK vs finite-difference | error ≈ 4e-7; ±0.16 ms/solve |
| Controller referensi (mengikuti jalur tervalidasi), 60 episode/misi | **60/60 sukses** di Misi 1, 2, 3 |
| Validasi skenario lolos percobaan pertama | M1 100%, M2 94%, M3 99% |
| Env eval vs env training (aksi & seed sama) | selisih observasi maksimum **0** |
| Transfer Misi 1→2 pada 301 observasi nyata | selisih aksi **0.0** |
| Ekspor ONNX + `verify_onnx` (observasi nyata, ketiga misi) | max\|diff\| ≤ 4.2e-7 |
| Smoke-run pipeline penuh M1→M2→M3 (training, transfer, ONNX, debug) | lolos |
| Skrip eval ×3 (headless), CSV, plot replay | lolos |
| Entry point `python train_mission1.py` dari foldernya; error jelas bila checkpoint misi sebelumnya belum ada | lolos |
| **Training Misi 1 sungguhan, hyperparameter asli, 300k langkah** | success eval 0% → 20% → 50% → 70% (threshold 10 cm); curriculum naik ke 5 cm pada ~280k langkah; 5 mm belum tercapai (belum waktunya) |

## 4. Belum tervalidasi — mohon perhatian
1. **Konvergensi penuh** (6 juta langkah Misi 1, 4 juta Misi 2, 6 juta Misi 3) belum dijalankan; sandbox hanya 1 CPU. Pada run 300k langkah, ±15–20% episode berakhir *self-collision*; ini wajar di awal, tetapi pantau `reasons/self_collision_rate` di TensorBoard.
2. **Viewer MuJoCo interaktif belum diuji** (tidak ada layar). Yang teruji: seluruh logika marker (T1, T2, garis, rintangan, zona aman, jejak EE) memakai `MjvScene` nyata, serta jalur headless. `launch_passive` baru akan benar-benar dijalankan di mesin Anda (macOS: `mjpython`).
3. **Hyperparameter & bobot reward** adalah titik awal. Jika success Misi 1 stagnan di level 5 mm, pertimbangkan menaikkan `timesteps` atau `check_every_n_episodes`, bukan mengubah struktur reward.
4. Di Misi 3, rintangan yang terpangkas tidak terlihat oleh policy. Skenario dijamin punya **satu** jalur bebas tabrakan, bukan bahwa semua jalur aman.



Berikut saran tambahan, urut dari yang paling berdampak. Saran nomor 1–3 berdasar temuan nyata dari pengujian. Sisanya pertimbangan umum yang belum saya buktikan di proyek ini.

Prioritas tinggi
1. Fitur resume training. Saat ini training selalu mulai dari nol. Untuk run berjam-jam, 
    resume dari checkpoint (termasuk VecNormalize dan curriculum_state.json) akan menghemat banyak waktu bila proses terputus.
2. Menekan self-collision. Di run 300k langkah, 15–20% episode berakhir self-collision. Kalau tetap tinggi setelah jutaan langkah,
    pertimbangkan penalti jarak-dekat antar link (mirip potensial rintangan di Misi 3), bukan hanya terminasi saat kontak.
3. Jaring pengaman pemilihan model. Saat curriculum baru naik level, success rate bisa turun sementara. 
4. Menyimpan checkpoint terbaik per level curriculum akan membantu bila level terakhir ternyata tidak stabil.

Prioritas menengah
1. Beberapa seed. Jalankan 2–3 seed per misi dan bandingkan kurvanya. Satu run bisa kebetulan bagus atau jelek.
2. Evaluasi yang lebih ketat. Tambah metrik seperti success rate per jarak target, per sisi workspace (azimuth), 
    dan per massa payload. Dengan begitu kelemahan spesifik terlihat, bukan hanya angka rata-rata.
3. Uji robustness lebih luas. Domain randomization saat ini hanya payload, gravitasi, dan derau awal. 
    Bila tujuan akhirnya robot nyata, tambahkan delay aksi, friksi sendi, dan derau sensor yang lebih besar.
4. Penyetelan hyperparameter. Setelah run pertama, coba variasi ent_coef, ukuran jaringan, dan n_steps. 
    Sebaiknya hanya setelah Anda melihat kurva baseline.

Prioritas rendah / tergantung tujuan
1. Lintasan Misi 3 yang lebih sulit. Rintangan bergerak, lebih banyak rintangan, atau konstelasi lebih rapat. Tambahkan hanya setelah Misi 3 stabil pada skenario saat ini.
2. Tes otomatis. Kumpulkan uji yang saya pakai (controller referensi 60/60, env eval = env training, transfer persis) menjadi satu folder tests/ agar mudah dijalankan ulang setelah Anda mengubah kode.
3. Uji di Windows/macOS asli, serta uji viewer interaktif, karena keduanya belum bisa saya verifikasi.
4. Persiapan sim-to-real (bila relevan): konversi aksi ke perintah robot nyata, batas kecepatan/percepatan, dan pengujian ONNX di runtime target.

Saran paling bernilai untuk tahap awal adalah nomor 1 dan 4. Mulailah dengan satu run Misi 1, lihat kurvanya, lalu putuskan sisanya berdasarkan apa yang benar-benar terlihat, bukan menambah fitur di muka.