#!/usr/bin/env python3
"""
YOLO Detection Recorder — optimise memoire (video 1080p+)
Appsink → AI externe (ONNX Runtime) → Appsrc

Optimisations memoire vs version precedente :
  - Queues limitees a 2 frames max (vs 30)
  - Pas de copie numpy inutile pour le display
  - Resolution de travail reduite (WORK_WIDTH) avant traitement
  - Une seule copie frame pour l'IA, pas de double buffer

Usage:
    python3 yolo_record.py
    python3 yolo_record.py output.mp4
    python3 yolo_record.py output.mp4 input.mp4
"""

import sys
import os
import time
import threading
import queue
import subprocess
import numpy as np
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

try:
    import cv2
except ImportError:
    print("[ERREUR] pip install opencv-python"); sys.exit(1)
try:
    import onnxruntime as ort
except ImportError:
    print("[ERREUR] pip install onnxruntime"); sys.exit(1)

# === CONFIGURATION ===
BASE_DIR       = "/home/marvin/Documents/onnx_detection/onnx_detection_gpu"
DEFAULT_VIDEO  = f"{BASE_DIR}/build/test-video_2.mp4"
DEFAULT_MODEL  = f"{BASE_DIR}/yolo11n_320.onnx"
DEFAULT_OUTPUT = f"{BASE_DIR}/output_app.mp4"
PLUGIN_PATH    = f"{BASE_DIR}/gst-template/builddir/gst-plugin"
ONNX_LIB_PATH  = "/opt/onnxruntime/lib"

CPU_THREADS    = 4
CONF_THRESHOLD = 0.45
NMS_THRESHOLD  = 0.45
INPUT_SIZE     = 320  # 320 recommande pour 1080p (moins de memoire + plus rapide)
H264_QP        = 20

# ── Resolution de travail interne ────────────────────────────────────────────
# Reduit la resolution AVANT le tee pour economiser la memoire
# La sortie est upscalee a la resolution source avant encodage
# 0 = pas de reduction (utilise la resolution native)
# 960 = travail en 960x540 pour du 1080p (gain memoire x4)
# 640 = travail en 640x360 pour du 1080p (gain memoire x9)
WORK_WIDTH = 960

MOTION_COMPENSATION = True

os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = ONNX_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

Gst.init(None)

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


# ─────────────────────────────────────────────────────────────────────────────
#  Resolution de travail (ratio conserve, multiple de 2)
# ─────────────────────────────────────────────────────────────────────────────
def compute_work_size(src_w, src_h, work_w):
    if work_w <= 0 or work_w >= src_w:
        return src_w, src_h
    work_h = int(src_h * work_w / src_w)
    return (work_w & ~1), (work_h & ~1)


# ─────────────────────────────────────────────────────────────────────────────
#  Tracker leger
# ─────────────────────────────────────────────────────────────────────────────
def iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter+1e-6)

class FrameTracker:
    def __init__(self, fw, fh):
        self.fw=fw; self.fh=fh
        self._lock=threading.Lock()
        self._tracks=[]; self._det_frame=-1

    def update(self, detections, frame_idx):
        with self._lock:
            new_boxes=[{"box":[x,y,x+w,y+h],"conf":conf,"cls":cls}
                       for (x,y,w,h),conf,cls in detections]
            updated=[]; used=set()
            for nb in new_boxes:
                best_iou=0.2; best_idx=-1
                for i,tr in enumerate(self._tracks):
                    if i in used or tr["cls"]!=nb["cls"]: continue
                    s=iou(nb["box"],tr["box"])
                    if s>best_iou: best_iou=s; best_idx=i
                if best_idx>=0:
                    old=self._tracks[best_idx]
                    df=frame_idx-old["frame"]
                    vel=[(nb["box"][k]-old["box"][k])/max(df,1) for k in range(4)]
                    updated.append({"box":nb["box"],"vel":vel,
                                    "conf":nb["conf"],"cls":nb["cls"],
                                    "frame":frame_idx,"prev":old["box"]})
                    used.add(best_idx)
                else:
                    updated.append({"box":nb["box"],"vel":[0,0,0,0],
                                    "conf":nb["conf"],"cls":nb["cls"],
                                    "frame":frame_idx,"prev":None})
            self._tracks=updated; self._det_frame=frame_idx

    def get_boxes(self, display_frame_idx):
        with self._lock:
            if not self._tracks: return []
            delta=display_frame_idx-self._det_frame
            result=[]
            for tr in self._tracks:
                if MOTION_COMPENSATION and tr["prev"] is not None and delta>0:
                    box=[int(tr["box"][k]+tr["vel"][k]*delta*0.3) for k in range(4)]
                else:
                    box=[int(v) for v in tr["box"]]
                box[0]=max(0,min(self.fw-1,box[0])); box[1]=max(0,min(self.fh-1,box[1]))
                box[2]=max(0,min(self.fw,box[2]));   box[3]=max(0,min(self.fh,box[3]))
                result.append((box,tr["conf"],tr["cls"]))
            return result


# ─────────────────────────────────────────────────────────────────────────────
#  AI
# ─────────────────────────────────────────────────────────────────────────────
class YoloInference:
    def __init__(self, model_path):
        opts=ort.SessionOptions()
        opts.intra_op_num_threads=CPU_THREADS
        opts.inter_op_num_threads=CPU_THREADS
        opts.execution_mode=ort.ExecutionMode.ORT_PARALLEL
        self.sess=ort.InferenceSession(model_path,sess_options=opts,
                      providers=["CPUExecutionProvider"])
        self.in_name=self.sess.get_inputs()[0].name
        self._lock=threading.Lock(); self._count=0; self._sum_ms=0.0

    def infer(self, frame_bgr):
        h,w=frame_bgr.shape[:2]
        resized=cv2.resize(frame_bgr,(INPUT_SIZE,INPUT_SIZE))
        rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        tensor=np.expand_dims(np.transpose(rgb.astype(np.float32)/255.0,(2,0,1)),0)
        t0=time.perf_counter()
        outputs=self.sess.run(None,{self.in_name:tensor})
        ms=(time.perf_counter()-t0)*1000
        with self._lock: self._count+=1; self._sum_ms+=ms
        return self._postprocess(outputs,w,h)

    def _postprocess(self, outputs, fw, fh):
        preds=outputs[0][0]; xs,ys=fw/INPUT_SIZE,fh/INPUT_SIZE
        boxes,confs,ids=[],[],[]
        for i in range(preds.shape[1]):
            scores=preds[4:,i]; cls_id=int(np.argmax(scores)); conf=float(scores[cls_id])
            if conf<CONF_THRESHOLD: continue
            cx,cy,bw,bh=preds[0,i],preds[1,i],preds[2,i],preds[3,i]
            boxes.append([int((cx-bw/2)*xs),int((cy-bh/2)*ys),int(bw*xs),int(bh*ys)])
            confs.append(conf); ids.append(cls_id)
        if not boxes: return []
        idx=cv2.dnn.NMSBoxes(boxes,confs,CONF_THRESHOLD,NMS_THRESHOLD)
        flat=idx.flatten() if hasattr(idx,'flatten') else idx
        return [(boxes[i],confs[i],ids[i]) for i in flat]

    def draw(self, frame, boxes):
        for box,conf,cls in boxes:
            x1,y1,x2,y2=box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            label=f"{COCO_CLASSES[cls]} {conf:.2f}"
            sz,bl=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(frame,(x1,y1-sz[1]-bl),(x1+sz[0],y1),(0,255,0),-1)
            cv2.putText(frame,label,(x1,y1-bl),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    def avg_fps(self):
        with self._lock:
            return 1000.0/(self._sum_ms/self._count) if self._count>0 else 0.0

    def count(self):
        with self._lock: return self._count


# ─────────────────────────────────────────────────────────────────────────────
#  Info video robuste
# ─────────────────────────────────────────────────────────────────────────────
def get_video_info(path):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height,r_frame_rate,nb_frames,duration",
             "-of","csv=p=0", path],
            capture_output=True, text=True, timeout=10)
        parts=r.stdout.strip().split(",")
        w,h=int(parts[0]),int(parts[1])
        num,den=parts[2].split("/")
        fps=round(int(num)/int(den))
        total=0
        if len(parts)>3 and parts[3].strip().isdigit():
            total=int(parts[3])
        if total==0 and len(parts)>4:
            try: total=int(float(parts[4])*fps)
            except: pass
        if total==0:
            r2=subprocess.run(["ffprobe","-v","error",
                "-show_entries","format=duration","-of","csv=p=0",path],
                capture_output=True,text=True,timeout=10)
            try: total=int(float(r2.stdout.strip())*fps)
            except: pass
        return w,h,fps,total
    except Exception as e:
        print(f"[WARN] fallback ({e})")
        return 1280,720,25,0


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, model_path, output_path):
    src_w, src_h, fps, total = get_video_info(video_path)
    work_w, work_h = compute_work_size(src_w, src_h, WORK_WIDTH)
    frame_duration_ns = Gst.SECOND // fps

    # Memoire estimee par frame en MB
    mem_src  = src_w  * src_h  * 3 / 1e6
    mem_work = work_w * work_h * 3 / 1e6
    # Avec queue de 2 frames + copie IA : ~6 frames en memoire max
    mem_total = mem_work * 6

    print("=" * 62)
    print("  YOLO — Appsink → AI → Appsrc (optimise memoire)")
    print("=" * 62)
    print(f"  Entree        : {video_path}")
    print(f"  Modele        : {model_path}")
    print(f"  Sortie        : {output_path}")
    print(f"  Source        : {src_w}x{src_h} @ {fps}fps  ({total} frames)")
    print(f"  Travail       : {work_w}x{work_h}  ({mem_work:.1f} MB/frame)")
    print(f"  Memoire ~max  : {mem_total:.0f} MB frames video")
    print(f"  CPU threads   : {CPU_THREADS}")
    print(f"  INPUT_SIZE    : {INPUT_SIZE}px")
    print(f"  MC            : {'active' if MOTION_COMPENSATION else 'desactive'}")
    print("=" * 62)
    print("  Chargement modele ONNX...")
    ai      = YoloInference(model_path)
    tracker = FrameTracker(work_w, work_h)
    print(f"  OK — {CPU_THREADS} threads ORT\n")

    # Queues tres petites pour limiter la memoire
    # max 2 frames en attente = ~2 * mem_work MB
    display_q     = queue.Queue(maxsize=2)
    infer_q       = queue.Queue(maxsize=1)
    src_eos       = threading.Event()
    stop_evt      = threading.Event()
    stats         = {"written":0,"t0":None}
    frame_idx_ctr = [0]

    # ── Pipeline destination : travail en work_w x work_h,
    #    upscale vers src_w x src_h avant encodage ──────────────────────────
    dst_pipeline = Gst.parse_launch(
        "appsrc name=src format=time is-live=false ! "
        "videoconvert ! videoscale ! "
        f"video/x-raw,width={src_w},height={src_h} ! "
        f"x264enc speed-preset=fast quantizer={H264_QP} ! "
        "h264parse ! mp4mux ! "
        f"filesink location={output_path} sync=false"
    )
    appsrc = dst_pipeline.get_by_name("src")
    appsrc.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={work_w},height={work_h},framerate={fps}/1"
    ))
    dst_pipeline.get_bus().add_signal_watch()
    def on_dst_msg(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err,dbg=msg.parse_error()
            print(f"\n[Display] {err.message}\n{dbg}")
    dst_pipeline.get_bus().connect("message", on_dst_msg)

    # ── Pipeline source : decode → reduit a work_w x work_h → appsink ─────
    src_pipeline = Gst.parse_launch(
        f"filesrc location={video_path} ! "
        f"decodebin name=dec "
        f"dec. ! queue ! audioconvert ! fakesink "
        f"dec. ! queue max-size-buffers=2 ! videoconvert ! videoscale ! "
        f"video/x-raw,format=BGR,width={work_w},height={work_h} ! "
        # max-buffers=2 : backpressure sur la source si on est en retard
        f"appsink name=sink emit-signals=true sync=true "
        f"max-buffers=2 drop=false"
    )
    appsink = src_pipeline.get_by_name("sink")

    def on_frame(sink):
        sample = sink.emit("pull-sample")
        if not sample: return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            # Pas de .copy() inutile — on copie une seule fois
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape(
                work_h, work_w, 3).copy()
            buf.unmap(info)
            fidx = frame_idx_ctr[0]
            frame_idx_ctr[0] += 1

            # Bloquer si display est plein (backpressure memoire)
            display_q.put((frame, fidx))

            # IA : frame partagee, pas de copie supplementaire
            # (l'IA lit, le display ecrit apres → ok car sequential dans infer_q)
            try:
                infer_q.put_nowait((frame, fidx))
            except queue.Full:
                pass   # IA occupee, on saute cette frame pour l'inference
        return Gst.FlowReturn.OK

    appsink.connect("new-sample", on_frame)

    src_bus = src_pipeline.get_bus()
    src_bus.add_signal_watch()
    def on_src_msg(bus, msg):
        if msg.type == Gst.MessageType.EOS:
            display_q.put(None); src_eos.set()
        elif msg.type == Gst.MessageType.ERROR:
            err,_=msg.parse_error()
            print(f"\n[Source] {err.message}")
            display_q.put(None); src_eos.set()
    src_bus.connect("message", on_src_msg)

    # ── Thread AI ─────────────────────────────────────────────────────────────
    def ai_thread():
        while not stop_evt.is_set():
            try:
                item = infer_q.get(timeout=1.0)
            except queue.Empty:
                if src_eos.is_set(): break
                continue
            if item is None: break
            frame, fidx = item
            dets = ai.infer(frame)
            tracker.update(dets, fidx)
            del dets   # liberer explicitement

    # ── Thread Display ─────────────────────────────────────────────────────────
    display_done = threading.Event()
    def display_thread():
        out_idx = 0
        while True:
            item = display_q.get()
            if item is None: break
            frame, fidx = item

            boxes = tracker.get_boxes(fidx)
            ai.draw(frame, boxes)

            ai_fps = ai.avg_fps()
            txt    = f"IA: {ai_fps:.1f} fps"
            cv2.putText(frame,txt,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
            cv2.putText(frame,txt,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,80),2)

            pts = out_idx * frame_duration_ns
            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            buf.pts=pts; buf.dts=pts; buf.duration=frame_duration_ns
            appsrc.emit("push-buffer", buf)
            del frame   # liberer apres push

            stats["written"] += 1
            out_idx += 1

            elapsed  = time.monotonic()-stats["t0"]
            if total > 0:
                pct = stats["written"]/total*100
                bar = "#"*int(25*pct/100)+"-"*(25-int(25*pct/100))
                pct_str = f"{pct:5.1f}%"
            else:
                pos = out_idx%25
                bar = "-"*pos+"#"+"-"*(24-pos)
                pct_str = f"{stats['written']:5d}f"
            realtime = stats["written"]/fps
            speed = realtime/elapsed if elapsed>0 else 0
            print(f"\r  [{bar}] {pct_str}  {stats['written']}/{total if total>0 else '?'}"
                  f"  |  IA:{ai_fps:.1f}fps  vitesse:{speed:.1f}x",
                  end="",flush=True)

        appsrc.emit("end-of-stream")
        display_done.set()

    # ── Demarrage ─────────────────────────────────────────────────────────────
    print("  Traitement...\n")
    stats["t0"] = time.monotonic()

    dst_pipeline.set_state(Gst.State.PLAYING)
    src_pipeline.set_state(Gst.State.PLAYING)

    t_ai   = threading.Thread(target=ai_thread,      daemon=True)
    t_disp = threading.Thread(target=display_thread, daemon=True)
    t_ai.start()
    t_disp.start()

    loop        = GLib.MainLoop()
    loop_thread = threading.Thread(target=loop.run, daemon=True)
    loop_thread.start()

    try:
        src_eos.wait()
        display_done.wait(timeout=120)
    except KeyboardInterrupt:
        print("\n  Interrompu.")
        display_q.put(None)

    stop_evt.set()
    loop.quit()
    time.sleep(0.5)
    src_pipeline.set_state(Gst.State.NULL)
    dst_pipeline.set_state(Gst.State.NULL)

    elapsed = time.monotonic()-stats["t0"]
    print(f"\n\n  Termine ! → {output_path}")
    print(f"  Frames ecrites  : {stats['written']}/{total if total>0 else '?'}")
    print(f"  Frames IA       : {ai.count()}")
    print(f"  FPS IA moyen    : {ai.avg_fps():.1f}")
    print(f"  Temps total     : {elapsed:.1f}s\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Mode batch
# ─────────────────────────────────────────────────────────────────────────────
def process_batch(video_list):
    for i, vpath in enumerate(video_list):
        if not os.path.exists(vpath): print(f"[SKIP] {vpath}"); continue
        base=os.path.splitext(os.path.basename(vpath))[0]
        out=os.path.join(os.path.dirname(vpath),f"{base}_yolo.mp4")
        print(f"\n[{i+1}/{len(video_list)}] {os.path.basename(vpath)}")
        run(vpath,DEFAULT_MODEL,out)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)>1 and all(a.endswith(".mp4") for a in args):
        process_batch(args)
    else:
        output = args[0] if args else DEFAULT_OUTPUT
        video  = args[1] if len(args)>1 else DEFAULT_VIDEO
        if not os.path.exists(video):
            print(f"[ERREUR] Video introuvable : {video}"); sys.exit(1)
        if not os.path.exists(DEFAULT_MODEL):
            print(f"[ERREUR] Modele introuvable : {DEFAULT_MODEL}"); sys.exit(1)
        run(video, DEFAULT_MODEL, output)