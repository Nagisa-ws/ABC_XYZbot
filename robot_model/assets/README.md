# robot_model/assets

Mesh yang direferensikan oleh `ur10e_eval.xml` (model **evaluasi**; mesh hanya visual, collision memakai primitif yang
identik dengan model training). Model **training** (`ur10e_train.xml`, `scene_train.xml`) tidak memakai mesh sama
sekali, jadi folder ini hanya dibutuhkan untuk tampilan MuJoCo viewer.

| Berkas | Dipakai oleh |
|---|---|
| `base_0/1.obj`, `shoulder_0/1/2.obj`, `upperarm_0/1/2/3.obj`, `forearm_0/1/2/3.obj`, `wrist1_0/1/2.obj`, `wrist2_0/1/2.obj`, `wrist3.obj` | lengan UR10e |
| `Servo-v1.stl`, `slide-left-v1.stl`, `slide-right-v3.stl` | gripper |
| `Pineaple.STL` | tidak direferensikan oleh XML mana pun di proyek ini |

## Sumber dan lisensi

### Lengan UR10e (`*.obj`)
Berasal dari **MuJoCo Menagerie** (Google DeepMind), direktori `universal_robots_ur10e`:
<https://github.com/google-deepmind/mujoco_menagerie>

Lisensi mesh ini mengikuti berkas `LICENSE` pada direktori tersebut. Salin berkas itu ke
`robot_model/assets/LICENSE-ur10e.txt` dan pertahankan pemberitahuan hak cipta aslinya. Model XML di folder induk
(`ur10e_train.xml`, `ur10e_eval.xml`, `scene_*.xml`) kemungkinan diturunkan dari `ur10e.xml` Menagerie dengan
perubahan (kompensasi gravitasi, geom collision primitif); bila benar, atribusi yang sama berlaku untuknya.

### Gripper (`Servo-v1.stl`, `slide-left-v1.stl`, `slide-right-v3.stl`)
Dirancang sendiri oleh pemilik repo ini. Mesh gripper dirilis di bawah lisensi yang sama dengan kode proyek (lihat `LICENSE` di akar repo); ubah baris ini bila Anda ingin lisensi berbeda untuk desain CAD.

Bila mesh tidak dapat didistribusikan, hapus folder ini dari repo: training tetap berjalan, hanya viewer evaluasi yang
membutuhkan mesh.
