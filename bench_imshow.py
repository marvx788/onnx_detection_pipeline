"""
Benchmark rapide : mesure le coût réel de cv2.resize + cv2.imshow
sur la machine cible, pour différentes résolutions.
Lance avec : python3 bench_imshow.py
"""
import cv2, numpy as np, time

SCALES   = [0.25, 0.50, 0.75, 1.0]
SRC_W, SRC_H = 1280, 720
N_FRAMES = 30

print("=" * 52)
print("  Benchmark cv2.imshow (pure timing)")
print("=" * 52)
frame_src = np.random.randint(0, 255, (SRC_H, SRC_W, 3), dtype=np.uint8)

for scale in SCALES:
    w = int(SRC_W * scale)
    h = int(SRC_H * scale)
    win = f"bench_{scale}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)

    # Warmup
    f = cv2.resize(frame_src, (w, h))
    cv2.imshow(win, f)
    cv2.waitKey(1)

    # Mesure resize seul
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        cv2.resize(frame_src, (w, h))
    resize_ms = (time.perf_counter() - t0) / N_FRAMES * 1000

    # Mesure imshow seul (frame déjà resizée)
    f = cv2.resize(frame_src, (w, h))
    t0 = time.perf_counter()
    for _ in range(N_FRAMES):
        cv2.imshow(win, f)
        cv2.waitKey(1)
    imshow_ms = (time.perf_counter() - t0) / N_FRAMES * 1000

    total_ms  = resize_ms + imshow_ms
    max_fps   = 1000 / total_ms if total_ms > 0 else 999

    print(f"  {int(scale*100):3d}%  {w:4d}x{h:3d} | "
          f"resize:{resize_ms:5.1f}ms  imshow+waitKey:{imshow_ms:6.1f}ms  "
          f"→ total:{total_ms:6.1f}ms  max:{max_fps:5.1f}fps")

cv2.destroyAllWindows()
print("=" * 52)