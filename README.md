# YOLO Detection — MJPEG Stream (C++)

Détection d'objets en temps réel (YOLO + OpenVINO + GStreamer + OpenCV) avec
prédicteur cinématique frame-par-frame et streaming MJPEG via serveur HTTP
intégré.

> **⚠️ Migration** : ce projet utilisait auparavant un script Python
> (`yolo_detection_recorder.py`, génération de vidéo annotée en post-traitement).
> Il est désormais remplacé par une implémentation **C++** optimisée pour
> Raspberry Pi 4, avec streaming live plutôt que génération de fichier vidéo.
> Le script Python n'est plus maintenu.

---

## Principe

- L'IA (OpenVINO) tourne en arrière-plan à fréquence variable (skip adaptatif
  selon le temps d'inférence mesuré).
- Un tracker interne fait correspondre chaque nouvelle détection IA à un objet
  suivi (matching par IoU + distance de centre).
- Dès qu'un objet a reçu ≥ 2 détections IA, sa vitesse est calculée et chaque
  frame d'affichage recalcule sa position prédite par extrapolation
  cinématique — aucune détection IA n'est nécessaire à chaque frame affichée.
- Le flux annoté est encodé en JPEG et diffusé en MJPEG sur un serveur HTTP
  intégré (pas de dépendance à un serveur externe).

---

## Requirements

### Système
```bash
sudo apt install build-essential pkg-config \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libopencv-dev \
    ffmpeg
```

### OpenVINO
Le runtime OpenVINO (C API) doit être installé et accessible au moment de la
compilation et de l'exécution (headers `openvino/c/openvino.h` + libs).
Voir la documentation officielle Intel pour l'installation sur ARM64 /
Raspberry Pi.

---

## Compilation

```bash
g++ -O3 -std=c++17 yolo_detection_stream.cpp -o yolo_detection_stream \
    $(pkg-config --cflags --libs opencv4) \
    $(pkg-config --cflags --libs gstreamer-1.0) \
    -lopenvino_c -lpthread
```

> Adapte le nom du fichier source et les chemins d'include/lib OpenVINO selon
> ton installation (`-I`, `-L` si besoin).

---

## Project Structure

```
onnx_detection_pipeline/
├── yolo_detection_stream.cpp     # programme principal (C++)
├── yolo11n_256.onnx              # modèle YOLO (ignoré par git)
├── test-video.mp4                # vidéo de test (ignorée par git)
└── README.md
```

---

## Usage

### Vidéo par défaut (chemins définis dans le code)
```bash
./yolo_detection_stream
```

### Spécifier une vidéo ou une caméra
```bash
./yolo_detection_stream /chemin/vers/video.mp4 /chemin/vers/model.onnx
./yolo_detection_stream /dev/video0 /chemin/vers/model.onnx
```

Le programme détecte automatiquement si l'entrée est une caméra (chemin
commençant par `/dev/video`) ou un fichier vidéo, et adapte le pipeline
GStreamer en conséquence (`v4l2src` vs `filesrc + decodebin`).

### Visualiser le flux
Une fois lancé, le flux MJPEG est accessible sur :
```
http://<ip-du-raspberry>:8080/stream
```
ou directement `http://<ip-du-raspberry>:8080/` pour une page HTML minimale
qui l'affiche.

---

## Configuration

Principaux paramètres en haut du fichier source (`static const`, nécessitent
une recompilation pour être modifiés) :

| Variable | Défaut | Description |
|---|---|---|
| `INPUT_SIZE` | `256` | Résolution d'entrée du modèle YOLO |
| `DISPLAY_SCALE` | `1.0` | Facteur d'échelle d'affichage (1.0 = pas de resize) |
| `JPEG_QUALITY` | `70` | Qualité d'encodage JPEG du flux MJPEG |
| `HTTP_PORT` | `8080` | Port du serveur MJPEG intégré |
| `CONF_KEEP` | `0.25` | Seuil de confiance minimum pour garder une détection |
| `NMS_THRESHOLD` | `0.45` | Seuil IoU pour la suppression non-maximale |
| `DISPLAY_FPS` | `25.0` | FPS d'affichage/streaming fixe (voir note ci-dessous) |
| `MC_IOU_MATCH` | `0.25` | Seuil IoU pour le matching tracker IA→objet |
| `TRACKER_MAX_AGE` | `3` | Frames IA manquées avant suppression d'un objet suivi |
| `SIZE_EMA_ALPHA` | `0.3` | Lissage exponentiel des tailles de boîtes |
| `MAX_OBJECTS` | `64` | Nombre max d'objets suivis simultanément |
| `TARGET_AI_MS` | `120.0` | Temps d'inférence cible pour le skip adaptatif |
| `MAX_SKIP` | `8` | Skip maximum entre deux inférences IA |
| `AI_EMA_ALPHA` | `0.2` | Lissage exponentiel du temps d'inférence mesuré |

> **`DISPLAY_FPS`** pilote à la fois la cadence d'affichage/streaming **et**
> la précision du prédicteur cinématique (conversion frames ↔ temps réel pour
> le calcul de vitesse). Elle doit correspondre au FPS réel de la source
> (caméra ou vidéo) pour que les prédictions de trajectoire restent justes —
> ce n'est pas une valeur purement cosmétique.

### Profiling

Le profiling (temps moyen par étape : capture, resize+enqueue IA, prédiction,
dessin, encodage JPEG) est activé par défaut via la macro `ENABLE_PROFILING`
en haut du fichier. Il s'affiche sur `stderr` toutes les ~5 secondes. Mettre
`#define ENABLE_PROFILING 0` pour le désactiver (recompilation nécessaire).

---

## Tester avec une caméra simulée (sans caméra physique)

Pour tester le pipeline avec le même chemin de code que celui utilisé en
production avec une vraie caméra (branche `v4l2src` du pipeline GStreamer),
sans disposer d'une caméra physique, on peut utiliser **v4l2loopback** pour
créer un faux device `/dev/videoX` et y injecter une vidéo mp4 en temps réel.

### 1. Installation

```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils raspberrypi-kernel-headers
```

> Sur Raspberry Pi OS, il faut impérativement les headers **spécifiques
> Raspberry Pi** (`raspberrypi-kernel-headers`), pas les headers Debian
> génériques (`linux-headers-*`), sous peine d'échec de compilation DKMS.
> Vérifier la correspondance de version avec `uname -r` et
> `ls /lib/modules/$(uname -r)/build`.

### 2. Charger le module (persistant au redémarrage)

Pour éviter de devoir relancer `modprobe` à chaque boot :

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=30 card_label="FakeCam" exclusive_caps=1' \
    | sudo tee /etc/modprobe.d/v4l2loopback.conf
sudo reboot
```

Après reboot, vérifier :
```bash
v4l2-ctl --list-devices
# → doit afficher "FakeCam" sur /dev/video30
```

> Le numéro de device (`video_nr`) doit être choisi libre : sur Raspberry Pi,
> les devices `bcm2835-codec` et `bcm2835-isp` occupent déjà plusieurs
> numéros bas (10-23 typiquement) — vérifier avec `v4l2-ctl --list-devices`
> avant de fixer `video_nr`.

### 3. Injecter une vidéo dans le faux device

```bash
ffmpeg -re -stream_loop -1 -i test-video.mp4 \
    -vf "scale=640:360" \
    -pix_fmt yuyv422 -f v4l2 /dev/video30
```

- `-re` : lecture au rythme natif de la vidéo (simule une source live, sinon
  ffmpeg pousse les frames plus vite que le framerate réel).
- `-stream_loop -1` : boucle infinie.
- Laisser tourner cette commande dans un terminal séparé pendant les tests.

### 4. Lancer le programme sur le faux device

```bash
./yolo_detection_stream /dev/video30 yolo11n_256.onnx
```

### Limite de ce test

Le flux simulé n'a pas la contrainte physique d'un vrai capteur caméra : selon
le buffering du pipeline (`queue max-size-buffers=N`, `sync=false`), des
rafales de frames peuvent ponctuellement pousser le FPS d'affichage mesuré
au-dessus du FPS réel de la source (25fps ici) avant de se restabiliser. Ce
n'est pas représentatif d'une vraie caméra (dont le débit est physiquement
plafonné) — réduire le buffering (`max-size-buffers=1 leaky=downstream`,
`sync=true`) rapproche le comportement du réel si besoin de validation plus
stricte.

---

## Obtenir le modèle YOLO

```bash
pip install ultralytics --break-system-packages
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=256)"
```
Place le fichier `.onnx` généré dans le dossier du projet (adapter
`DEFAULT_MODEL` dans le code si besoin, ou passer le chemin en argument).