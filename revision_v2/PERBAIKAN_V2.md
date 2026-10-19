# Perbaikan v2: gerak lebih halus dan langsung, resume training, grafik yang lengkap

Dokumen ini menjelaskan **apa yang diubah, mengapa (dengan pengukuran), apa yang sudah terbukti, dan apa yang belum**.
Semua perubahan reward diatur dari config; bobot 0 (atau kunci dihapus) = perilaku lama, jadi model dan config lama
tetap bisa dipakai.

> **Perlu training ulang Misi 1 → 2 → 3.** Model lama tidak bisa diperbaiki dengan fine-tuning (bagian 4.3).

![Perbandingan desain lama dan baru](docs/perbaikan_v2_perbandingan.png)

---------------------------------------------------------------------------------------------------------------------

## 1. Diagnosis model lama (100 skenario tetap per misi, seed 20000+)

Dibandingkan dengan controller referensi garis lurus (`make_predict_fn_scripted`) pada skenario yang sama:

| | Misi 1 | Misi 2 | Misi 3 | Referensi lurus |
|---|---|---|---|---|
| Penyimpangan fase A dari garis lurus | 166 mm | 165 mm | 132 mm | 4 mm |
| Panjang jalur fase A ÷ jarak lurus | 1,26 | 1,18 | 1,16 | ~0,98 |
| Rata-rata \|aksi rotasi\| (maks 1) | 0,91 | 0,90 | 0,89 | 0 |
| Total putaran gripper / episode | 115° | 119° | 132° | 13–18° |
| Percepatan ujung gripper (RMS) | 4,6 | 5,1 | 5,1 m/s² | 1,0–2,0 |
| Sukses | 81% (5 mm) | 96% | 80% | 95–100% |

Penyebab yang terukur:

1. **Jalur berliuk hampir gratis.** Reward progres `w·(d_prev − d)` bersifat teleskopik: total imbalannya sama untuk
   jalur lurus maupun melengkung. Uji keseimbangan reward (gaya gerak sintetis): jalur busur dengan penyimpangan
   ±100 mm hanya rugi −5 poin per episode dibanding jalur lurus.
2. **Rotasi gripper tanpa tujuan.** Misi 1–3 tidak punya tugas orientasi, jadi 3 dari 6 dimensi aksi tidak mendapat
   sinyal (rotasi jenuh hanya rugi −0,5 poin). Policy menekan rotasi sampai batas kerucut 30° dan gripper
   "mengembara" di sana, ikut menggeser lintasan. Di Misi 3, controller lurus yang diberi rotasi jenuh turun dari
   100% ke 83% sukses, dan versi busurnya hanya 27% (menabrak rintangan).
3. **Penalti kehalusan tidak berefek.** 0,05 × ||Δa||² ≈ −0,006 per langkah, jauh di bawah reward progres.
4. **Celah "bunuh diri".** Menabrak bernilai tetap (−10), sedangkan meneruskan episode yang buruk bisa jauh lebih
   negatif. Begitu penalti per langkah dinaikkan, policy belajar SENGAJA menabrak lantai (22 dari 30 episode eval).
5. **Misi 3 sebenarnya dapat diselesaikan.** Controller lurus yang mengikuti jalur referensi tervalidasi: 100% sukses,
   0 tabrakan. Policy RL tertahan di ±80% (baseline lama di sandbox juga berakhir di 77%) karena jalurnya menyimpang
   (70% tabrakan terjadi di fase A) dan rotasinya jenuh.
6. **Wilayah kerja Misi 1–2 sempit.** Hanya mencakup ±10% titik awal T1 Misi 3; 6 dari 8 timeout Misi 3 ada di T1 di
   luar wilayah itu. (Perbaikannya tersedia sebagai opsi, lihat 2.2; belum dijadikan default.)

## 2. Perubahan

### 2.1 Reward (`shared/envs/base_arm_env.py`, `Mission*/configs/domain_randomization.yaml`)

| Kunci config | v2 | Arti |
|---|---|---|
| `lateral_motion_penalty` | 20 | R = −w · \|\|komponen langkah TCP yang menyamping dari arah ke target\|\| (m); gerak maju ke target tidak dikenai biaya |
| `orientation_penalty` | 1,0 | R = −w · θ², θ = kemiringan gripper terhadap pose nominal (rad) |
| `rotation_effort_penalty` | 0,05 | R = −w · \|\|a_rot\|\|² |
| `smoothness_penalty` | 0,1 (lama 0,05) | R = −w · min(\|\|a_t − a_(t−1)\|\|², `smoothness_cap`) |
| `smoothness_cap` | 1,0 | batas per langkah: derau eksplorasi awal (std 1 → \|\|Δa\|\|² ≈ 12) tidak boleh mendominasi |
| `failure_penalty_per_remaining_step` | 1,0 | penalti gagal (kontak/tabrakan) = penalti dasar − c · sisa langkah: menabrak tidak pernah lebih murah daripada meneruskan episode |
| `path_length_penalty` | 0 | alternatif yang juga menghukum gerak maju; tersedia untuk eksperimen |

`ppo.log_std_init: -0.5` di Misi 1 (std eksplorasi awal 0,6; default SB3 1,0 membuat banyak aksi terpotong di ±1).

Bobot dipilih lewat uji keseimbangan reward, lalu disetel dengan training pilot. Versi pertama (lateral 40, kehalusan
0,3, tanpa batas) "memajaki" derau eksplorasi terlalu berat: policy ragu-ragu dan belajar jauh lebih lambat
(34% sukses training pada 266 ribu langkah, lama 81% pada 223 ribu).

### 2.2 Kurikulum wilayah kerja (OPSIONAL, default mati)

`adaptive.workspace_schedule` di `Mission1/configs/ppo_config.yaml` (contoh disertakan sebagai komentar): tiga level
pertama berambang 10 cm dengan wilayah target sempit → tengah → lebar (r 0,25–0,85 m, h −0,03–0,74 m, mencakup T1
Misi 3), lalu ambang diketatkan. Env mendapat `set_workspace()`, callback mencatat `curriculum/workspace_r_max`.

Mengapa tidak default: di wilayah lebar Misi 1 belajar ±2× lebih lambat, dan pada ambang 2 cm policy pilot memburuk
(eval 23% → 10%: mendekat, melewati target, lalu tidak kembali). Pada ambang 4 cm di wilayah lebar, v2 justru unggul
(71% vs lama 40%). Bila dicoba: perlebar juga wilayah Misi 2 dan boleh hapus `reach_r_max`/`reach_z_max` di kurikulum
Misi 3; anggarkan langkah Misi 1 lebih banyak dan pantau `curriculum/level`.

### 2.3 Training

* **Resume** (`resume:` di setiap `ppo_config.yaml`): lanjutkan training yang terhenti atau tambah langkah. Model +
  VecNormalize + level kurikulum dipulihkan, transfer dan pemanasan critic dilewati, `best_model` lama hanya diganti
  bila hasil baru lebih baik, hitungan langkah dan kurva TensorBoard bersambung. `run.timesteps` = langkah TAMBAHAN.
  (Teruji di sandbox: model 1.001.472 langkah dan level kurikulum 5 dipulihkan dengan benar.)
* Traceback `BrokenPipeError`/`EOFError` setelah Ctrl+C tidak ditampilkan lagi (galat saat menutup worker diabaikan;
  semua berkas sudah tersimpan sebelumnya).

### 2.4 Grafik dan metrik

* `training_curves.png` kini 3×3. Panel **Alasan terminasi** diperbaiki (tag `reasons/*` tidak terbaca sehingga panel
  selalu kosong), ditambah baris **kualitas lintasan**: rasio jalur fase A, putaran & kemiringan gripper, perubahan aksi
  per langkah. Panel tanpa data diberi keterangan, bukan kotak kosong.
* TensorBoard: `rollout/path_ratio_A`, `rollout/rot_travel_deg`, `rollout/tilt_mean_deg`, `rollout/action_jerk`
  (dan versi `eval/`).
* `replay_*.png`: panel keempat **aksi & kemiringan gripper per langkah**; panel cross-track menjelaskan bila fase B
  tidak pernah dimulai (dulu kosong).
* `robustness.png`: panel "jumlah rintangan" hanya di Misi 3; Misi 1–2 menampilkan skala gravitasi yang diacak
  (dulu panel itu selalu 0).
* **Baru: `path_quality.png`** setelah training: policy vs controller garis lurus pada skenario yang sama.

### 2.5 Yang ikut dibereskan

Berkas yang sebelumnya hanya ada di zip Colab kini ada di repo: `shared/evaluation/video_render.py`,
`shared/evaluation/drake_replay.py` (multi-skenario), `shared/exporter/versioning.py` + impor lazy (evaluasi tidak
memuat torch/SB3; ini yang mencegah crash di Colab).

## 3. Kompatibilitas

* Observasi (57 dimensi) dan aksi (6 dimensi) **tidak berubah**: ONNX, controller hibrida, evaluasi viewer, Colab dan
  Drake bekerja dengan model lama maupun baru. Dinamika env tidak berubah, jadi controller LQR tetap valid.
* Dengan config lama, env v2 identik bit-per-bit dengan env lama (diuji: observasi, reward, terminasi; 3 misi).
* Kembali ke perilaku lama: set bobot di 2.1 ke 0, `smoothness_penalty` 0,05, hapus `smoothness_cap`.

## 4. Bukti dari training pilot

Sandbox 2 CPU, 2 env, **1 juta langkah** per run (training Anda 6–10 juta, 4 env), satu seed per varian.

### 4.1 Misi 1: desain lama vs v2, anggaran dan seed sama, 100 skenario identik

| | Ambang 2 cm: lama | **2 cm: v2** | Ambang 4 cm: lama | **4 cm: v2** |
|---|---|---|---|---|
| Sukses | 70% | **71%** | 90% | **92%** |
| Self-collision | 29% | **2%** | 10% | **2%** |
| Timeout | 1% | 27% | 0% | 6% |
| Rasio jalur (episode sukses) | 1,36 | **1,02** | 1,20 | **1,00** |
| Penyimpangan dari garis lurus (sukses) | 140 mm | **69 mm** | 131 mm | **84 mm** |
| Lebih lurus pada skenario yang sama | – | **100% dari 50** | – | **99% dari 82** |
| \|aksi rotasi\| / kemiringan maks | 0,86 / 30° | **0,11 / 9°** | 0,85 / 30° | **0,11 / 7°** |
| Putaran gripper / episode | 92° | **39°** | 85° | **30°** |
| Perubahan aksi posisi / langkah | 0,45 | **0,08** | 0,42 | **0,09** |
| Percepatan ujung gripper (RMS) | 13,6 | **3,5** | 12,3 | **3,6** m/s² |

Kecepatan belajar setara: level ambang 4 cm dicapai pada 353 ribu langkah (lama 347 ribu) dan 2 cm pada 638 ribu
(lama 578 ribu). Eval di ambang 2 cm: v2 53–63% pada 700 ribu–1 juta langkah, lama 37–47%.

Kegagalan v2 bergeser dari tabrakan dengan badan sendiri ke **timeout**: policy mendekat dengan kecepatan penuh,
melewati target beberapa cm, lalu tidak kembali. Ini bagian yang perlu dipantau pada training penuh.

### 4.2 Misi 2 → 3: belum terbukti

Rantai penuh v2 tidak bisa diuji di sandbox. Model Misi 1 berumur 1 juta langkah, **lama maupun v2**, memberi titik
awal Misi 2 sebesar 0% (diukur lewat proses transfer sungguhan), karena belum cukup presisi untuk ambang 2 cm Misi 2.
Rantai yang bermakna butuh Misi 1 terlatih penuh. Harapan untuk Misi 3 bertumpu pada mekanisme yang terukur (rotasi
jenuh dan jalur melengkung adalah penyebab langsung tabrakan; lihat 1.2 dan 1.5), bukan pada hasil training Misi 3 v2.

### 4.3 Reward v2 tidak bisa memperbaiki model lama

Misi 3 dari model Misi 2 lama, kurikulum lama, seed sama, 300 ribu langkah: reward lama 77–80% vs reward v2 77%;
putaran gripper malah naik (117° → 126°). Rata-rata aksi rotasi model lama sudah di luar ±1, sehingga semua sampel
aksi terpotong ke nilai yang sama dan penalti rotasi tidak memberi gradien. Perbaikan butuh training ulang dari Misi 1.

## 5. Risiko dan batasan

* Hasil Misi 2–3 v2 belum diukur (4.2). Bandingkan dengan model lama Anda sebelum mengganti model yang dipakai.
* Timeout karena *overshoot* di ambang presisi (4.1). Bila Misi 1 macet di satu level, tambah langkah dengan `resume`.
* Jalur v2 jauh lebih lurus tetapi belum selurus controller referensi (penyimpangan median 69–84 mm); controller
  hibrida tetap pilihan untuk ketelitian milimeter.
* Satu seed per varian; anggaran pilot jauh lebih pendek dari training Anda.

## 6. Cara training ulang

```bash
python Mission1/train_mission1.py      # 10 juta langkah (config)
python Mission2/train_mission2.py      # dari checkpoints/mission1/best_model.zip
python Mission3/train_mission3.py      # dari checkpoints/mission2/best_model.zip
tensorboard --logdir checkpoints
```

Cadangkan `checkpoints/` dan `models/` lama lebih dulu (training menimpa `best_model.zip`; ONNX mendapat versi baru).
Pantau di TensorBoard selain success rate: `rollout/path_ratio_A` (mendekati 1 = lurus), `rollout/rot_travel_deg`,
`rollout/action_jerk`, `reasons/self_collision_rate`, `curriculum/level`. Setelah training, lihat
`checkpoints/missionN/debug/path_quality.png` dan `training_curves.png`.
