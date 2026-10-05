# Controller hibrida (RL + LQR) untuk Misi 2 dan 3

## Apa ini
Model RL Anda tetap dipakai apa adanya (ONNX tidak berubah, tidak perlu training ulang). Di atasnya dipasang
controller LQR **untuk posisi** yang mengambil alih saat gripper mendekati T1 (jarak < 8 cm) atau saat fase
menyusuri T1->T2 dimulai. Aksi rotasi tetap dari RL. Controller hanya membaca observasi 57 angka yang sama.

## Pakai
```
python Mission3/benchmark_hybrid_mission3.py     # RL saja vs hibrida, ambang 2 cm dan 5 mm (skenario tetap)
python Mission2/benchmark_hybrid_mission2.py
```
Evaluasi viewer: di `Mission*/configs/eval_config.yaml` set `hybrid.enabled: true`
(dan, untuk presisi, `rollout.t1_reach_threshold` dan `rollout.success_pos_threshold` ke 0.005).
Deploy di luar proyek:
```python
from shared.control import make_hybrid_from_onnx
policy = make_hybrid_from_onnx("models/onnx_mission3_v1.onnx")   # parameter: shared/control/controller.yaml
policy.reset()                  # WAJIB di awal tiap episode (controller menyimpan riwayat posisi)
action = policy(obs_mentah)     # obs mentah 57-dim; normalisasi sudah ada di dalam ONNX
```
Yang dikirim bersama ONNX: folder `shared/control/` (+ `shared/envs/obs_layout.py` untuk indeks observasi).
Tidak ada dependensi baru (tidak butuh scipy).

## Hasil terukur (model Anda, skenario tetap seed 20000+, level 4 untuk Misi 3)
| | Ambang | RL saja | RL + LQR | Cross-track |
|---|---|---|---|---|
| Misi 3 (100) | 2 cm | 80,0% | 87,0% | 26,1 -> 1,4 mm |
| Misi 3 (100) | 5 mm | 4,0% | 81,0% | 21,3 -> 0,9 mm |
| Misi 2 (100) | 2 cm | 94,0% | 98,0% | 23,5 -> 2,2 mm |
| Misi 2 (100) | 5 mm | 7,0% | 97,0% | 27,2 -> 1,4 mm |
Eksperimen lebih besar (200 skenario Misi 3): 80,5% -> 85,5% (2 cm), 4,5% -> 81,5% (5 mm).
Pada ambang 5 mm RL saja nyaris selalu timeout: model dilatih untuk ambang 2 cm.

## Yang terukur tentang desain
- LQR vs hukum P sederhana: sukses hampir sama, tetapi cross-track LQR ~3x lebih kecil dan lebih cepat.
  LQR dengan bobot bawaan awal lebih buruk dari P; bobot sekarang hasil penalaan (70 kombinasi pada 40
  skenario Misi 2) dan divalidasi pada 60 skenario lain.
- Jarak serah-terima 4-15 cm: hasil praktis identik. Pencampuran 5-15 langkah: tidak berpengaruh.
- Derau sensor (pada tcp_pos, rel_target, cross_track_vec): cross-track 1,0 mm (tanpa derau) -> 2,6 mm (1 mm)
  -> 8,2 mm (5 mm); sukses tetap ~81%. Lonjakan aksi naik (1,2 -> 1,9): pada robot nyata dengan derau besar,
  aktuator bisa bergetar. Filter belum dibuat.

## Batasan (jujur)
- Controller TIDAK menghindari rintangan. Tabrakan tetap ~10-14% di Misi 3 (tidak berkurang, tidak bertambah).
- Controller TIDAK menggantikan RL di pendekatan jauh (garis lurus gagal pada skenario berjalur polar).
- Orientasi gripper (moncong sejajar jalur) BELUM dikerjakan. Gripper tetap miring ~24 derajat dari vertikal.
- Bobot ditala pada env Misi 2; diuji pada Misi 2 dan 3 dengan satu model Anda. Misi 1 tidak diuji.
- Model plant adalah perkiraan (R2 0,96-0,99; kopling antar sumbu hingga ~40%); umpan balik menutupi galatnya.
- Belum diuji: derau sensor nyata, hardware nyata, dan model selain milik Anda.
- `max_pos_delta` di env (0,05) harus sama dengan di controller.yaml; benchmark memeriksanya.


---------------------------------------------------------------------------------------------------
# Tambahan: penyejajaran moncong gripper dengan jalur (opsional, default MATI)

Tujuan: moncong gripper (sumbu z TCP) sejajar dengan arah jalur T1->T2, seperti ular yang membidik mangsa.
Aktifkan: `orientation.enabled: true` di `shared/control/controller.yaml`, atau `hybrid.orientation: true` di
`Mission*/configs/eval_config.yaml`. Benchmark selalu membandingkan tiga policy (RL saja / posisi / posisi+orientasi).

## Aturan (opsi 1): hanya untuk jalur MENURUN CURAM
Aktif hanya bila sudut jalur terhadap "lurus ke bawah" < 60 derajat. Pada jalur lain perilaku = hibrida posisi saja.
Alasan (terukur): pada jalur mendatar/menanjak, penyejajaran menurunkan sukses 11-18 poin (self-collision & kontak
lantai) karena controller orientasi tidak sadar tabrakan; pada jalur menurun curam tidak ada penurunan.

## Hasil (model Anda, skenario tetap seed 20000+, 100 skenario, ambang 2 cm / 5 mm)
| | RL saja | RL + LQR (posisi) | RL + LQR + orientasi | Jalur menurun curam: galat moncong |
|---|---|---|---|---|
| Misi 3, 2 cm | 80,0% | 87,0% | **89,0%** | 43,3 -> **6,7 deg**, 72% langkah < 10 deg (n=47) |
| Misi 3, 5 mm | 4,0% | 81,0% | **85,0%** | 43,2 -> **6,6 deg**, 73% < 10 deg |
| Misi 2, 2 cm | 94,0% | 98,0% | 98,0% | 41,0 -> **3,3 deg**, 95% < 10 deg (n=22) |
| Misi 2, 5 mm | 7,0% | 97,0% | 97,0% | 41,2 -> **3,3 deg**, 95% < 10 deg |
Selisih sukses orientasi vs posisi saja di Misi 3: 4 menang / 2 kalah (2 cm), 6 / 2 (5 mm) -> tidak lebih buruk;
belum diuji signifikansinya. Hanya sekitar 47% skenario Misi 3 dan 22% skenario Misi 2 yang jalurnya menurun curam.

## Cara kerja & syarat
- Posisi DAN orientasi diserahkan ke controller sejak 40 cm sebelum T1 (bukan 8 cm). Terukur: bila rotasi dimulai
  saat RL masih mengendalikan posisi, sukses jatuh (97,5% -> 77,5% pada 25 cm, 52,5% pada 60 cm): RL tidak toleran
  terhadap orientasi gripper yang berubah. Serah-terima dari 60 cm gagal (garis lurus gagal pada jalur berjalur polar).
- Controller orientasi melacak PERINTAH rotasinya sendiri dengan anti-windup (tanpa itu perintah berlari ~180 deg di
  depan orientasi aktual, IK melenceng, lengan menabrak lantai: terukur) dan menahan gerak sepanjang jalur sampai
  galat moncong < 10 deg (maks 80 langkah).
- Batas kerucut orientasi env (30 deg) HARUS dilonggarkan saat aktif: `policy.cone_request` -> `env.max_orient_dev`
  (dilakukan `bind_to_env`, dipakai benchmark & evaluasi viewer). Di robot nyata, antarmuka perintah harus menerima
  rentang orientasi lebih lebar pada tahap itu.

## Batasan (jujur)
- Jalur mendatar dan menanjak TIDAK disejajarkan (moncong tetap seperti hibrida posisi saja, miring ~24 deg dari vertikal).
- Aturan 60 derajat dipilih dari pengamatan per-arah pada 100 skenario (subset kecil: 19 jalur menanjak di Misi 3).
- Penyejajaran parsial untuk jalur lain, orientasi sadar-tabrakan, dan Misi 4 berbasis RL: belum dikerjakan/diuji.
- Satu model Anda, hanya di sandbox, tanpa derau sensor. Orientasi tidak sadar rintangan; pada jalur menurun curam
  tabrakan tidak bertambah (Misi 3: 10 -> 8 pada 2 cm), tetapi itu bukan jaminan.
