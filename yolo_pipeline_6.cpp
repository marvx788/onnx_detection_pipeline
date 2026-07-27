/*
 * YOLO Detection — MJPEG Stream + Prédicteur Cinématique Frame-par-Frame
 * Optimisé Raspberry Pi 4 (OpenCV 4.x + OpenVINO C API)
 *
 * Principe :
 *   - L'IA tourne en arrière-plan à fréquence variable (skip adaptatif)
 *   - Dès que l'IA a fourni >= 2 détections pour un objet, on calcule
 *     la vitesse V = (pos_n - pos_{n-1}) / DeltaT
 *     où DeltaT = n_frames_ecoulees * (1/DISPLAY_FPS)
 *   - Chaque frame d'affichage est VIERGE : on recalcule la prédiction
 *     depuis la dernière détection IA : pos = pos_n + V * k * dt_affichage
 *     avec k = nombre de frames depuis la dernière détection IA
 *   - AUCUNE persistance des prédictions entre frames
 *   - Le tracker sert uniquement au matching IA→objet (ID stable)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <csignal>
#include <cerrno>

#include <vector>
#include <string>
#include <algorithm>

#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* OpenCV C++ API (4.x) */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

/* OpenVINO C API */
#include <openvino/c/openvino.h>

/* ─── CONFIGURATION ──────────────────────────────────────────────────────── */
#define BASE_DIR       "/home/comvis/Documents/onnx_detection_pipeline"
static const char *DEFAULT_VIDEO = BASE_DIR "/test-video.mp4";
static const char *DEFAULT_MODEL = BASE_DIR "/yolo11n_256.onnx";

static const int    INPUT_SIZE      = 256;
static const float  DISPLAY_SCALE  = 1.5f;
static const int    JPEG_QUALITY   = 70;
static const int    HTTP_PORT      = 8080;

static const float  CONF_KEEP      = 0.25f;
static const float  NMS_THRESHOLD  = 0.45f;

static const double DISPLAY_FPS    = 25.0;   // FPS AFFICHAGE FIXE
static const double DT_DISPLAY      = 1.0 / DISPLAY_FPS; // durée d'une frame d'affichage

/* Tracking / prédiction cinématique */
static const float  MC_IOU_MATCH       = 0.25f;
static const int    TRACKER_MAX_AGE    = 3;    // frames IA manquées avant suppression

/* Lissage exponentiel pour les tailles de boîtes (évite les sauts drastiques) */
static const float  SIZE_EMA_ALPHA     = 0.3f;

static const int    MAX_OBJECTS    = 64;
static const int    MAX_CLASSES    = 80;

/* Skip adaptatif IA */
static const double TARGET_AI_MS   = 120.0;
static const int    MAX_SKIP       = 8;
static const double AI_EMA_ALPHA   = 0.2;

/* ─── CLASSES COCO ───────────────────────────────────────────────────────── */
static const char *COCO_CLASSES[80] = {
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
};

static cv::Scalar CLASS_COLORS[80];

static void init_colors() {
    uint64_t state = 42;
    for (int i = 0; i < 80; i++) {
        double ch[3];
        for (int c = 0; c < 3; c++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            ch[c] = 80.0 + (double)((state >> 33) % 150);
        }
        CLASS_COLORS[i] = cv::Scalar(ch[0], ch[1], ch[2]);
    }
}

/* ─── UTILITAIRES TEMPS ──────────────────────────────────────────────────── */
static double perf_counter() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_sec(double s) {
    if (s <= 0.0) return;
    struct timespec ts;
    ts.tv_sec  = (time_t)s;
    ts.tv_nsec = (long)((s - (double)(time_t)s) * 1e9);
    nanosleep(&ts, NULL);
}

/* ─── SKIP ADAPTATIF ─────────────────────────────────────────────────────── */
struct AdaptiveSkip {
    double ai_ms_ema = 100.0;
    double alpha = AI_EMA_ALPHA;
    int current_skip = 1;
    int frame_counter = 0;
    pthread_mutex_t lock;

    AdaptiveSkip() { pthread_mutex_init(&lock, NULL); }
    ~AdaptiveSkip() { pthread_mutex_destroy(&lock); }

    void update(double ai_ms) {
        pthread_mutex_lock(&lock);
        ai_ms_ema = alpha * ai_ms + (1.0 - alpha) * ai_ms_ema;
        double ideal_skip = ai_ms_ema * DISPLAY_FPS / 1000.0;
        current_skip = std::max(1, std::min(MAX_SKIP, (int)std::ceil(ideal_skip)));
        pthread_mutex_unlock(&lock);
    }

    bool should_process() {
        pthread_mutex_lock(&lock);
        frame_counter++;
        if (frame_counter >= current_skip) {
            frame_counter = 0;
            pthread_mutex_unlock(&lock);
            return true;
        }
        pthread_mutex_unlock(&lock);
        return false;
    }

    int get_skip() {
        pthread_mutex_lock(&lock);
        int s = current_skip;
        pthread_mutex_unlock(&lock);
        return s;
    }
};

/* ─── TYPES DÉTECTION ────────────────────────────────────────────────────── */
struct Box { float x, y, w, h; };

struct Detection {
    Box   box;
    float conf;
    int   cls_id;
};

/* ─── NMS ────────────────────────────────────────────────────────────────── */
static float iou_boxes(const Box &a, const Box &b) {
    float xi1 = std::max(a.x, b.x), yi1 = std::max(a.y, b.y);
    float xi2 = std::min(a.x+a.w, b.x+b.w), yi2 = std::min(a.y+a.h, b.y+b.h);
    float inter = std::max(0.f, xi2-xi1) * std::max(0.f, yi2-yi1);
    float uni   = a.w*a.h + b.w*b.h - inter;
    return uni > 0.f ? inter/uni : 0.f;
}

static int nms_detections(std::vector<Detection> &dets, float thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b){ return a.conf > b.conf; });
    std::vector<bool> sup(dets.size(), false);
    std::vector<Detection> kept;
    for (size_t i = 0; i < dets.size(); i++) {
        if (sup[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i+1; j < dets.size(); j++) {
            if (!sup[j] && dets[i].cls_id == dets[j].cls_id &&
                iou_boxes(dets[i].box, dets[j].box) > thresh)
                sup[j] = true;
        }
    }
    dets = kept;
    return (int)dets.size();
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  PRÉDICTEUR CINÉMATIQUE FRAME-PAR-FRAME
 *
 *  Chaque objet tracké stocke les deux dernières détections IA :
 *    - det_prev  : détection à la frame IA n-1
 *    - det_cur   : détection à la frame IA n
 *
 *  Vitesse estimée (pixels/frame d'affichage) :
 *    V_cx = (cx_n - cx_{n-1}) / DeltaFrames
 *    V_cy = (cy_n - cy_{n-1}) / DeltaFrames
 *    V_w  = (w_n  - w_{n-1})  / DeltaFrames
 *    V_h  = (h_n  - h_{n-1})  / DeltaFrames
 *  avec DeltaFrames = frame_idx_n - frame_idx_{n-1} (nombre de frames d'affichage)
 *
 *  Prédiction à la frame d'affichage k (k > frame_idx_n) :
 *    cx(k) = cx_n + V_cx * (k - frame_idx_n)
 *    cy(k) = cy_n + V_cy * (k - frame_idx_n)
 *    w(k)  = w_n  + V_w  * (k - frame_idx_n)
 *    h(k)  = h_n  + V_h  * (k - frame_idx_n)
 *
 *  Avant d'avoir 2 détections on ne prédit rien pour cet objet.
 *  Les tailles w,h sont lissées par EMA pour éviter les sauts.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 *  PRÉDICTEUR CINÉMATIQUE FRAME-PAR-FRAME  —  CORRIGÉ PAR FACTEUR D'ÉCHELLE
 *
 *  Chaque objet tracké stocke les deux dernières détections IA :
 *    - det_prev  : détection à la frame IA n-1
 *    - det_cur   : détection à la frame IA n
 *
 *  Vitesse estimée (pixels/frame d'affichage) :
 *    V_cx = (cx_n - cx_{n-1}) / DeltaFrames
 *    V_cy = (cy_n - cy_{n-1}) / DeltaFrames
 *    V_w  = (w_n  - w_{n-1})  / DeltaFrames
 *    V_h  = (h_n  - h_{n-1})  / DeltaFrames
 *  avec DeltaFrames = frame_idx_n - frame_idx_{n-1}
 *
 *  FACTEUR D'ÉCHELLE :
 *    Ln   = taille caractéristique à la détection n (ex: sqrt(w*h))
 *    Ln-1 = taille caractéristique à la détection n-1
 *    scale_factor = Ln / Ln-1
 *
 *  Prédiction à la frame d'affichage k (k > frame_idx_n) :
 *    cx(k) = cx_n + V_cx * dt * (Ln / Ln-1)        ← correction position
 *    cy(k) = cy_n + V_cy * dt * (Ln / Ln-1)
 *    w(k)  = w_n  + V_w  * dt * (Ln / Ln-1)        ← correction taille
 *    h(k)  = h_n  + V_h  * dt * (Ln / Ln-1)
 *
 *  Avant d'avoir 2 détections on ne prédit rien pour cet objet.
 * ═══════════════════════════════════════════════════════════════════════════ */

struct KinState {
    /* Détection courante (frame IA n) */
    float cx_cur, cy_cur, w_cur, h_cur;
    float conf_cur;
    int   frame_idx_cur;

    /* Détection précédente (frame IA n-1) */
    float cx_prev, cy_prev, w_prev, h_prev;
    int   frame_idx_prev;

    /* Facteurs d'échelle caractéristiques */
    float L_cur  = 0.0f;   // sqrt(w*h) de la détection courante
    float L_prev = 0.0f;   // sqrt(w*h) de la détection précédente

    bool has_two = false;
    bool initialized = false;

    /* Lissage des tailles */
    float w_smooth = 0, h_smooth = 0;
    bool  smooth_init = false;

    /* Ajoute une nouvelle détection confirmée par l'IA */
    void push(const Box &b, float conf, int frame_idx) {
        // Lissage exponentiel des tailles pour éviter les sauts
        if (!smooth_init) {
            w_smooth = b.w;
            h_smooth = b.h;
            smooth_init = true;
        } else {
            w_smooth = SIZE_EMA_ALPHA * b.w + (1.0f - SIZE_EMA_ALPHA) * w_smooth;
            h_smooth = SIZE_EMA_ALPHA * b.h + (1.0f - SIZE_EMA_ALPHA) * h_smooth;
        }

        // Calcul du facteur d'échelle caractéristique L = sqrt(w*h)
        float L_new = sqrtf(w_smooth * h_smooth);

        if (!initialized) {
            cx_cur = b.x + b.w/2;
            cy_cur = b.y + b.h/2;
            w_cur  = w_smooth;
            h_cur  = h_smooth;
            conf_cur = conf;
            frame_idx_cur = frame_idx;
            L_cur = L_new;
            initialized = true;
        } else {
            // Décale courant → précédent
            cx_prev = cx_cur; cy_prev = cy_cur;
            w_prev  = w_cur;  h_prev  = h_cur;
            frame_idx_prev = frame_idx_cur;
            L_prev = L_cur;

            // Nouvelle détection courante
            cx_cur = b.x + b.w/2;
            cy_cur = b.y + b.h/2;
            w_cur  = w_smooth;
            h_cur  = h_smooth;
            conf_cur = conf;
            frame_idx_cur = frame_idx;
            L_cur = L_new;
            has_two = true;
        }
    }

    /*
     * Calcule la boîte prédite pour la frame d'affichage frame_idx_now.
     * Retourne false si on n'a pas encore deux détections.
     * 
     * CORRECTION PAR FACTEUR D'ÉCHELLE :
     *   scale_factor = L_cur / L_prev
     *   Cela compense l'approche/éloignement de l'objet.
     */
    bool predict(int frame_idx_now, Box &out_box, float &out_conf) const {
        if (!has_two)
            return false;

        int delta_frames = frame_idx_cur - frame_idx_prev;
        if (delta_frames < 1)
            delta_frames = 1;

        int k = frame_idx_now - frame_idx_cur;
        if (k < 0)
            k = 0;

        // Limite de prédiction : on évite de prédire trop loin
        if (k > delta_frames * 2)
            return false;

        // Facteur d'échelle caractéristique
        // L = sqrt(w*h)  →  rapport Ln / Ln-1
        float scale_factor = 1.0f;
        if (L_prev > 1e-6f && L_cur > 1e-6f) {
            scale_factor = L_cur / L_prev;
            // Clamp pour éviter les explosions (objet très petit ou très grand)
            scale_factor = std::max(0.5f, std::min(2.0f, scale_factor));
        }

        float DeltaT = delta_frames / (float)DISPLAY_FPS;
        float dt = k / (float)DISPLAY_FPS;

        // Vitesse de base (pixels par seconde)
        float V_cx = (cx_cur - cx_prev) / DeltaT;
        float V_cy = (cy_cur - cy_prev) / DeltaT;
        float V_w  = (w_cur  - w_prev)  / DeltaT;
        float V_h  = (h_cur  - h_prev)  / DeltaT;

        // Prédiction avec correction d'échelle
        // La vitesse apparente est modulée par le changement de taille
        float pred_cx = cx_cur + V_cx * dt * scale_factor;
        float pred_cy = cy_cur + V_cy * dt * scale_factor;
        float pred_w  = std::max(1.f, w_cur + V_w * dt * scale_factor);
        float pred_h  = std::max(1.f, h_cur + V_h * dt * scale_factor);

        out_box.x = pred_cx - pred_w/2.f;
        out_box.y = pred_cy - pred_h/2.f;
        out_box.w = pred_w;
        out_box.h = pred_h;
        out_conf  = conf_cur;

        return true;
    }
};

/* ─── TRACKER INTERNE (matching IA → objet) ──────────────────────────────── */
struct TrackedObject {
    int         id;
    int         cls_id;
    cv::Scalar  color;
    int         consecutive_missed;
    int         age;
    KinState    kin;         // prédicteur cinématique frame-par-frame
    int         last_ai_frame_idx; // dernière frame d'affichage où l'IA a vu l'objet
    Box         last_box;    // boîte IA la plus récente (brute)
    float       last_conf;
};

struct Tracker {
    std::vector<TrackedObject> objects;
    int    max_age;
    float  iou_threshold;
    pthread_mutex_t lock;
    int    next_id = 0;

    Tracker(int max_age_, float iou_thr)
        : max_age(max_age_), iou_threshold(iou_thr)
    { pthread_mutex_init(&lock, NULL); }

    ~Tracker() { pthread_mutex_destroy(&lock); }
};

static float center_dist_norm(const Box &a, const Box &b) {
    float cx1 = a.x+a.w/2, cy1 = a.y+a.h/2;
    float cx2 = b.x+b.w/2, cy2 = b.y+b.h/2;
    float avg  = (a.w+a.h+b.w+b.h)/4.f + 1e-6f;
    float dx = cx1-cx2, dy = cy1-cy2;
    return sqrtf(dx*dx+dy*dy) / avg;
}

static void tracker_update(Tracker &t,
                            const std::vector<Detection> &dets,
                            int frame_idx) {
    pthread_mutex_lock(&t.lock);

    std::vector<bool> matched(t.objects.size(), false);
    for (const auto &d : dets) {

        int best_obj = -1;
        float best_score = -1.f;

        for (size_t oi = 0; oi < t.objects.size(); oi++) {

            if (
                matched[oi] ||
                t.objects[oi].cls_id != d.cls_id
            )
                continue;

            Box ref;
            float tmp_conf;

            if (
                !t.objects[oi]
                    .kin
                    .predict(
                        frame_idx,
                        ref,
                        tmp_conf
                    )
            ) {
                ref =
                    t.objects[oi]
                        .last_box;
            }

            float iou =
                iou_boxes(
                    d.box,
                    ref
                );

            float cdist =
                center_dist_norm(
                    d.box,
                    ref
                );

            float score =
                2.f * iou +
                2.f *
                std::max(
                    0.f,
                    1.f - cdist
                );

            if (
                score > best_score &&
                (
                    iou > 0.05f ||
                    cdist < 1.2f
                )
            ) {
                best_score =
                    score;

                best_obj =
                    oi;
            }
        }

        if (best_obj >= 0) {

            TrackedObject &obj =
                t.objects[best_obj];

            obj.last_box =
                d.box;

            obj.last_conf =
                d.conf;

            obj.last_ai_frame_idx =
                frame_idx;

            obj.consecutive_missed =
                0;

            obj.age++;

            obj.kin.push(
                d.box,
                d.conf,
                frame_idx
            );

            matched[best_obj] =
                true;

        } else if (
            (int)t.objects.size()
            < MAX_OBJECTS
        ) {

            TrackedObject obj{};

            obj.id =
                ++t.next_id;

            obj.cls_id =
                d.cls_id;

            obj.color =
                CLASS_COLORS[
                    d.cls_id % 80
                ];

            obj.last_box =
                d.box;

            obj.last_conf =
                d.conf;

            obj.last_ai_frame_idx =
                frame_idx;

            obj.age =
                1;

            obj.consecutive_missed =
                0;

            obj.kin.push(
                d.box,
                d.conf,
                frame_idx
            );

            t.objects.push_back(
                obj
            );

            matched.push_back(
                false
            );
        }
    }

    for (size_t oi = 0; oi < t.objects.size(); oi++) {
        if (!matched[oi]) {
            t.objects[oi].consecutive_missed++;
        }
    }

    t.objects.erase(
        std::remove_if(t.objects.begin(), t.objects.end(),
            [&](const TrackedObject &o) {
                return o.consecutive_missed >= t.max_age;
            }),
        t.objects.end());

    pthread_mutex_unlock(&t.lock);
}

/*
 * Retourne les boîtes prédites par le prédicteur cinématique
 * pour la frame d'affichage frame_idx_now.
 * Seuls les objets ayant >= 2 détections sont inclus.
 * Chaque frame est calculée indépendamment depuis la dernière détection IA.
 */
static std::vector<Detection> tracker_predict_kinematic(Tracker &t, int frame_idx_now) {
    pthread_mutex_lock(&t.lock);
    std::vector<Detection> result;

    for (auto &obj : t.objects) {
        Box   pred_box;
        float pred_conf;
        if (!obj.kin.predict(frame_idx_now, pred_box, pred_conf)) continue;

        result.push_back({pred_box, pred_conf, obj.cls_id});
    }

    if (result.size() >= 2)
        nms_detections(result, NMS_THRESHOLD);

    pthread_mutex_unlock(&t.lock);
    return result;
}

/* ─── INFÉRENCE OPENVINO ─────────────────────────────────────────────────── */
struct OVModel {
    ov_core_t              *core    = nullptr;
    ov_model_t             *model   = nullptr;
    ov_compiled_model_t    *compiled= nullptr;
    ov_infer_request_t     *req     = nullptr;
    pthread_mutex_t         lock;
    int    count  = 0;
    double sum_ms = 0.0;
    double last_ms = 0.0;

    OVModel() { pthread_mutex_init(&lock, NULL); }
    ~OVModel() { pthread_mutex_destroy(&lock); }
};

static bool ov_init(OVModel &m, const char *model_path) {
    if (ov_core_create(&m.core) != OK) {
        fprintf(stderr, "[OV] ov_core_create failed\n"); return false;
    }
    if (ov_core_read_model(m.core, model_path, NULL, &m.model) != OK) {
        fprintf(stderr, "[OV] read_model failed: %s\n", model_path); return false;
    }

    const char *device = "CPU";
    ov_available_devices_t devs;
    if (ov_core_get_available_devices(m.core, &devs) == OK) {
        for (size_t i = 0; i < devs.size; i++)
            if (strcmp(devs.devices[i], "GPU") == 0) { device = "GPU"; break; }
        ov_available_devices_free(&devs);
    }
    printf("  Device selected : %s\n", device);

    if (ov_core_compile_model(m.core, m.model, device, 0, &m.compiled) != OK) {
        fprintf(stderr, "[OV] compile_model failed\n"); return false;
    }
    if (ov_compiled_model_create_infer_request(m.compiled, &m.req) != OK) {
        fprintf(stderr, "[OV] create_infer_request failed\n"); return false;
    }
    return true;
}

static void ov_free(OVModel &m) {
    if (m.req)      ov_infer_request_free(m.req);
    if (m.compiled) ov_compiled_model_free(m.compiled);
    if (m.model)    ov_model_free(m.model);
    if (m.core)     ov_core_free(m.core);
}

static std::vector<Detection> ov_infer(OVModel &m, const cv::Mat &frame, double* out_ms) {
    int fw = frame.cols, fh = frame.rows;

    cv::Mat small;
    cv::resize(frame, small, cv::Size(INPUT_SIZE, INPUT_SIZE));
    cv::Mat rgb;
    cv::cvtColor(small, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);

    std::vector<float> tensor(3 * INPUT_SIZE * INPUT_SIZE);
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < INPUT_SIZE; r++)
            for (int col = 0; col < INPUT_SIZE; col++)
                tensor[c*INPUT_SIZE*INPUT_SIZE + r*INPUT_SIZE + col] =
                    rgb.at<cv::Vec3f>(r, col)[c];

    int64_t shape[] = {1, 3, (int64_t)INPUT_SIZE, (int64_t)INPUT_SIZE};
    ov_shape_t ov_shape = {4, shape};
    ov_tensor_t *ov_t = nullptr;
    ov_tensor_create_from_host_ptr(ov_element_type_e::F32, ov_shape,
                                   tensor.data(), &ov_t);
    ov_infer_request_set_input_tensor(m.req, ov_t);

    double t0 = perf_counter();
    ov_infer_request_infer(m.req);
    double ms = (perf_counter() - t0) * 1000.0;
    ov_tensor_free(ov_t);

    pthread_mutex_lock(&m.lock);
    m.count++; m.sum_ms += ms; m.last_ms = ms;
    pthread_mutex_unlock(&m.lock);

    if (out_ms) *out_ms = ms;

    ov_tensor_t *out_t = nullptr;
    ov_infer_request_get_output_tensor(m.req, &out_t);
    void *raw = nullptr;
    ov_tensor_data(out_t, &raw);
    const float *preds = (const float*)raw;

    ov_shape_t out_shape;
    ov_tensor_get_shape(out_t, &out_shape);
    int n_anchors = (int)out_shape.dims[2];

    float xs = (float)fw / INPUT_SIZE;
    float ys = (float)fh / INPUT_SIZE;

    std::vector<Detection> raw_dets;
    raw_dets.reserve(64);

    for (int i = 0; i < n_anchors; i++) {
        int   best_cls  = 0;
        float best_conf = preds[4 * n_anchors + i];
        for (int k = 1; k < 80; k++) {
            float s = preds[(4+k) * n_anchors + i];
            if (s > best_conf) { best_conf = s; best_cls = k; }
        }
        if (best_conf < CONF_KEEP) continue;
        float cx = preds[0*n_anchors+i], cy = preds[1*n_anchors+i];
        float bw = preds[2*n_anchors+i], bh = preds[3*n_anchors+i];
        raw_dets.push_back({{(cx-bw/2)*xs,(cy-bh/2)*ys,bw*xs,bh*ys},
                             best_conf, best_cls});
    }
    ov_tensor_free(out_t);

    nms_detections(raw_dets, NMS_THRESHOLD);
    return raw_dets;
}

/* ─── DESSIN ─────────────────────────────────────────────────────────────── */
static void draw_detections(cv::Mat &frame, const std::vector<Detection> &dets) {
    for (const auto &d : dets) {
        int x = (int)d.box.x, y = (int)d.box.y;
        int w = (int)d.box.w, h = (int)d.box.h;
        cv::Scalar color = CLASS_COLORS[d.cls_id % 80];
        cv::rectangle(frame, cv::Rect(x, y, w, h), color, 2, cv::LINE_AA);

        char label[80];
        snprintf(label, sizeof(label), "%s: %.0f%%",
                 COCO_CLASSES[d.cls_id % 80], d.conf * 100.f);

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                      0.55, 1, &baseline);
        int ly = std::max(y - 4, ts.height + 4);
        cv::rectangle(frame,
                      cv::Point(x, ly - ts.height - baseline - 4),
                      cv::Point(x + ts.width + 4, ly),
                      color, cv::FILLED);
        cv::putText(frame, label, cv::Point(x+2, ly-baseline-2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }
}

/* ─── JPEG PARTAGÉ ───────────────────────────────────────────────────────── */
static pthread_mutex_t  g_jpeg_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   g_jpeg_cond = PTHREAD_COND_INITIALIZER;
static std::vector<uint8_t> g_jpeg_buf;
static int              g_jpeg_seq  = 0;

/* ─── SERVEUR MJPEG ──────────────────────────────────────────────────────── */
static volatile sig_atomic_t g_stop = 0;

struct ClientArg { int fd; };

static void *client_thread(void *arg) {
    int fd = ((ClientArg*)arg)->fd;
    free(arg);

    char buf[2048] = {};
    ssize_t n = recv(fd, buf, sizeof(buf)-1, 0);
    if (n <= 0) { close(fd); return NULL; }

    bool is_stream = strstr(buf, "GET /stream") != NULL;
    bool is_root   = strstr(buf, "GET / ")      != NULL
                  || strstr(buf, "GET /\r")     != NULL;

    if (is_root) {
        const char *html =
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            "<!DOCTYPE html><html><head><title>YOLO MJPEG</title>"
            "<style>body{margin:0;background:#111;display:flex;"
            "justify-content:center;align-items:center;height:100vh}"
            "img{max-width:100%;max-height:100vh}</style>"
            "</head><body><img src=\"/stream\"></body></html>\r\n";
        send(fd, html, strlen(html), 0);
    } else if (is_stream) {
        const char *hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache\r\n\r\n";
        send(fd, hdr, strlen(hdr), 0);

        int last_sent = -1;
        while (!g_stop) {
            std::vector<uint8_t> jpg_copy;

            pthread_mutex_lock(&g_jpeg_lock);
            if (g_jpeg_seq == last_sent) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_sec += 1;
                pthread_cond_timedwait(&g_jpeg_cond, &g_jpeg_lock, &ts);
            }
            if (!g_jpeg_buf.empty() && g_jpeg_seq != last_sent) {
                jpg_copy   = g_jpeg_buf;
                last_sent  = g_jpeg_seq;
            }
            pthread_mutex_unlock(&g_jpeg_lock);

            if (jpg_copy.empty()) continue;

            char part[128];
            int  plen = snprintf(part, sizeof(part),
                                 "--frame\r\nContent-Type: image/jpeg\r\n\r\n");
            if (send(fd, part, plen, MSG_NOSIGNAL) < 0 ||
                send(fd, jpg_copy.data(), jpg_copy.size(), MSG_NOSIGNAL) < 0 ||
                send(fd, "\r\n", 2, MSG_NOSIGNAL) < 0)
                break;
        }
    } else {
        const char *r404 = "HTTP/1.1 404 Not Found\r\n\r\n";
        send(fd, r404, strlen(r404), 0);
    }
    close(fd);
    return NULL;
}

static void *http_server_thread(void *arg) {
    int port = *(int*)arg;
    free(arg);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 16);

    while (!g_stop) {
        struct sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int fd = accept(srv, (struct sockaddr*)&cli, &cli_len);
        if (fd < 0) continue;
        ClientArg *ca = (ClientArg*)malloc(sizeof(*ca));
        ca->fd = fd;
        pthread_t t;
        pthread_create(&t, NULL, client_thread, ca);
        pthread_detach(t);
    }
    close(srv);
    return NULL;
}

/* ─── THREAD IA ──────────────────────────────────────────────────────────── */
struct AIItem { cv::Mat frame; int frame_idx; };

struct AIQueue {
    AIItem          slot;
    bool            has_item = false;
    pthread_mutex_t lock;
    pthread_cond_t  cond_consumer;

    AIQueue() {
        pthread_mutex_init(&lock, NULL);
        pthread_cond_init(&cond_consumer, NULL);
    }
    ~AIQueue() {
        pthread_mutex_destroy(&lock);
        pthread_cond_destroy(&cond_consumer);
    }
};

struct AIThreadArg {
    AIQueue       *ai_q;
    OVModel       *ov;
    Tracker       *tracker;
    AdaptiveSkip  *skip;
    bool           running;
};

static void *ai_thread_fn(void *arg) {
    AIThreadArg *a = (AIThreadArg*)arg;
    while (a->running) {
        AIItem item;
        bool got = false;

        pthread_mutex_lock(&a->ai_q->lock);
        if (!a->ai_q->has_item) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 200000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&a->ai_q->cond_consumer, &a->ai_q->lock, &ts);
        }
        if (a->ai_q->has_item) {
            item = std::move(a->ai_q->slot);
            a->ai_q->has_item = false;
            got = true;
        }
        pthread_mutex_unlock(&a->ai_q->lock);

        if (!got) continue;

        double infer_ms = 0;
        std::vector<Detection> dets = ov_infer(*a->ov, item.frame, &infer_ms);

        a->skip->update(infer_ms);

        // Remise à l'échelle display
        if (DISPLAY_SCALE != 1.0f)
            for (auto &d : dets) {
                d.box.x *= DISPLAY_SCALE; d.box.y *= DISPLAY_SCALE;
                d.box.w *= DISPLAY_SCALE; d.box.h *= DISPLAY_SCALE;
            }

        // Mise à jour du tracker avec le numéro de frame d'affichage
        tracker_update(*a->tracker, dets, item.frame_idx);
    }
    return NULL;
}

/* ─── THREAD ENCODAGE JPEG ──────────────────────────────────────────────── */
struct EncodeSlot {
    cv::Mat         frame;
    bool            has_frame = false;
    pthread_mutex_t lock;
    EncodeSlot() { pthread_mutex_init(&lock, NULL); }
    ~EncodeSlot() { pthread_mutex_destroy(&lock); }
};

struct EncodeThreadArg {
    EncodeSlot *slot;
    bool        running;
};

static void *encode_thread_fn(void *arg) {
    EncodeThreadArg *a = (EncodeThreadArg*)arg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY};
    std::vector<uint8_t> jpeg_buf;

    while (a->running) {
        cv::Mat frame;
        bool got = false;

        pthread_mutex_lock(&a->slot->lock);
        if (a->slot->has_frame) {
            frame = std::move(a->slot->frame);
            a->slot->has_frame = false;
            got = true;
        }
        pthread_mutex_unlock(&a->slot->lock);

        if (!got) { usleep(5000); continue; }

        if (
            frame.empty()
        ) {
            usleep(5000);
            continue;
        }

        try {
            cv::imencode(
                ".jpg",
                frame,
                jpeg_buf,
                params
            );
        }
        catch (...) {
            continue;
        }

        pthread_mutex_lock(&g_jpeg_lock);
        g_jpeg_buf = jpeg_buf;
        g_jpeg_seq++;
        pthread_cond_broadcast(&g_jpeg_cond);
        pthread_mutex_unlock(&g_jpeg_lock);
    }
    return NULL;
}

/* ─── SIGNAL ──────────────────────────────────────────────────────────────── */
static void handle_sigint(int) {
    g_stop = 1;
    pthread_cond_broadcast(&g_jpeg_cond);
}

/* ─── PIPELINE PRINCIPAL ─────────────────────────────────────────────────── */
static bool run_pipeline(const char *video_path, const char *model_path) {
    init_colors();
    signal(SIGINT, handle_sigint);

   char pipeline[1024];

    /* Détection auto : caméra ou fichier */
    bool is_camera =
        strncmp(video_path, "/dev/video", 10) == 0;

    if (is_camera) {

        snprintf(
        pipeline,
        sizeof(pipeline),

        "v4l2src device=%s ! "
        "queue max-size-buffers=3 ! "
        "videoscale ! "
        "video/x-raw,width=640,height=360 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink sync=false drop=true",

        video_path
        );

    } else {

        snprintf(
        pipeline,
        sizeof(pipeline),

        "filesrc location=%s ! "
        "queue max-size-buffers=3 ! "
        "decodebin ! "
        "queue max-size-buffers=3 ! "
        "videoscale ! "
        "video/x-raw,width=640,height=360 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink sync=false drop=true",

        video_path
        );
    }

    /* OUVERTURE RÉELLE */
    cv::VideoCapture cap(
        pipeline,
        cv::CAP_GSTREAMER
    );
    if (!cap.isOpened()) {
        fprintf(stderr, "[ERREUR] Impossible d'ouvrir : %s\n", video_path);
        return false;
    }
    int    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps    = cap.get(cv::CAP_PROP_FPS);
    int    total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    int    disp_w = (int)(width  * DISPLAY_SCALE);
    int    disp_h = (int)(height * DISPLAY_SCALE);
    double source_duration =
    (total > 0 && fps > 0)
    ? total / fps
    : 0.0;

    printf("================================================\n");
    printf("  YOLO — MJPEG Stream (Prédicteur Cinématique)\n");
    printf("================================================\n");
    printf("  Input   : %s\n", video_path);
    printf("  Source  : %dx%d @ %.1ffps (%d frames, %.1fs)\n",
           width, height, fps, total, source_duration);
    printf("  Display : %dx%d @ %.1ffps (FIXE)\n", disp_w, disp_h, DISPLAY_FPS);
    printf("  Prédicteur : cinématique pur frame-par-frame\n");
    printf("  Chargement modèle ONNX...\n");

    OVModel ov;
    if (!ov_init(ov, model_path)) return false;

    Tracker tracker(TRACKER_MAX_AGE, MC_IOU_MATCH);

    AIQueue    ai_q;
    EncodeSlot enc_slot;
    AdaptiveSkip skip_controller;

    AIThreadArg ai_arg { &ai_q, &ov, &tracker, &skip_controller, true };
    pthread_t t_ai;
    pthread_create(&t_ai, NULL, ai_thread_fn, &ai_arg);

    EncodeThreadArg enc_arg { &enc_slot, true };
    pthread_t t_enc;
    pthread_create(&t_enc, NULL, encode_thread_fn, &enc_arg);

    int *port_ptr = (int*)malloc(sizeof(int));
    *port_ptr = HTTP_PORT;
    pthread_t t_http;
    pthread_create(&t_http, NULL, http_server_thread, port_ptr);
    pthread_detach(t_http);

    printf("  Serveur MJPEG démarré sur le port %d\n", HTTP_PORT);
    printf("  En attente de 2 détections IA avant affichage des boîtes...\n\n");

    int    displayed = 0, ai_processed = 0;
    double t_program_start = perf_counter();
    int    frame_idx = 0;
    double last_disp_t = 0.0, disp_fps = 0.0;
    double disp_fps_acc = 0.0; int disp_fps_n = 0;
    double t_real_start = perf_counter();

    while (!g_stop) {
        cv::Mat raw;
        cap >> raw;

        double actual_fps = disp_fps > 1.0 ? disp_fps : DISPLAY_FPS;

        if (actual_fps < DISPLAY_FPS * 0.95) {

            int skip_frames =
                (int)(DISPLAY_FPS / actual_fps) - 1;

            for (int i = 0; i < skip_frames; i++) {
                cv::Mat dummy;

                if (!cap.read(dummy))
                    break;

                frame_idx++;
            }
        }

        if (raw.empty()) break;

        frame_idx++; // numéro de frame d'affichage (1, 2, 3, ...)

        /* ─── ENVOYER À L'IA SI C'EST LE TOUR ─── */
        bool do_ai = skip_controller.should_process();
        if (do_ai) {
            pthread_mutex_lock(&ai_q.lock);
            ai_q.slot = { raw.clone(), frame_idx };
            ai_q.has_item = true;
            pthread_cond_signal(&ai_q.cond_consumer);
            pthread_mutex_unlock(&ai_q.lock);
            ai_processed++;
        }


        /* ─── FRAME D'AFFICHAGE ─── */
        cv::Mat disp;

        if (DISPLAY_SCALE != 1.0f) {
            cv::resize(
                raw,
                disp,
                cv::Size(),
                DISPLAY_SCALE,
                DISPLAY_SCALE
            );
        } else {
            disp = raw.clone();
        }

        /* ─── PRÉDICTION CINÉMATIQUE PURE FRAME-PAR-FRAME ───
        *
        * Chaque frame est recalculée indépendamment depuis la
        * dernière détection IA.
        */
        std::vector<Detection> pred_dets =
            tracker_predict_kinematic(
                tracker,
                frame_idx
            );

        draw_detections(
            disp,
            pred_dets
        );

        /* ─── OSD ─── */
        double t_now = perf_counter();
        if (last_disp_t > 0.0) {
            double dt_disp = t_now - last_disp_t;
            if (dt_disp > 1e-9) { disp_fps_acc += 1.0/dt_disp; disp_fps_n++; }
        }
        last_disp_t = t_now;
        if (disp_fps_n > 0) disp_fps = disp_fps_acc / disp_fps_n;

        double ai_fps = 0.0;
        pthread_mutex_lock(&ov.lock);
        if (ov.count > 0) ai_fps = ov.count / (ov.sum_ms / 1000.0);
        pthread_mutex_unlock(&ov.lock);

        /* Compte objets */
        int n_tracked = 0, n_predictable = 0;
        pthread_mutex_lock(&tracker.lock);
        n_tracked = (int)tracker.objects.size();
        for (auto &o : tracker.objects) {
            if (o.kin.has_two) n_predictable++;
        }
        pthread_mutex_unlock(&tracker.lock);

        int current_skip = skip_controller.get_skip();

        char osd[256];
        snprintf(osd, sizeof(osd),
                 "Disp:%.1ffps  AI:%.1ffps  skip:%d  Obj:%d(%dP)",
                 disp_fps, ai_fps, current_skip,
                 n_tracked, n_predictable);
        cv::putText(disp, osd, cv::Point(10,28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0,0,0), 3, cv::LINE_AA);
        cv::putText(disp, osd, cv::Point(10,28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0,220,80), 2, cv::LINE_AA);

        /* ─── ENVOYER À L'ENCODEUR JPEG ─── */
        pthread_mutex_lock(&enc_slot.lock);
        enc_slot.frame = disp;
        enc_slot.has_frame = true;
        pthread_mutex_unlock(&enc_slot.lock);

        displayed++;

        /* ─── PROGRESS BAR ─── */
        float pct =
            (total > 0)
            ? (100.f * frame_idx / total)
            : 0.f;
        int bar_n = (int)(25 * pct / 100);
        char bar[27] = {};
        for (int i=0; i<25; i++) bar[i] = i < bar_n ? '#' : '-';
        printf("\r  [%s] %5.1f%%  F:%d  skip:%d  Disp:%5.1ffps  AI:%5.1ffps  Obj:%d(%dP)",
               bar, (double)pct, frame_idx, current_skip,
               disp_fps, ai_fps, n_tracked, n_predictable);
        fflush(stdout);

        /* ─── SYNCHRONISATION FPS FIXE ─── */
        double next_display_time = t_program_start + displayed / DISPLAY_FPS;
        double sleep_t = next_display_time - perf_counter();

        if (sleep_t > 0.0) {
            sleep_sec(sleep_t);
        } else if (sleep_t < -0.1) {
            t_program_start += 0.05;
        }
    }

    printf("\n\n  Finalisation...\n");
    g_stop = 1;
    ai_arg.running = enc_arg.running = false;
    pthread_cond_broadcast(&ai_q.cond_consumer);
    pthread_cond_broadcast(&g_jpeg_cond);

    pthread_join(t_ai,  NULL);
    pthread_join(t_enc, NULL);

    cap.release();
    ov_free(ov);

    double elapsed = perf_counter() - t_real_start;
    printf("  Frames lues      : %d\n", frame_idx);
    printf("  Frames affichées : %d\n", displayed);
    printf("  Frames IA traités: %d\n", ai_processed);
    printf("  Durée écoulée    : %.1fs\n", elapsed);
    double real_source_duration =
    (frame_idx > 0 && fps > 0)
    ? frame_idx / fps
    : 0.0;

    printf("  Durée source     : %.1fs\n", real_source_duration);
    printf("  FPS affiché moy. : %.1f\n", displayed / elapsed);
    printf("  Terminé !\n");
    return true;
}

int main(int argc, char *argv[]) {
    const char *video = argc > 1 ? argv[1] : DEFAULT_VIDEO;
    const char *model = argc > 2 ? argv[2] : DEFAULT_MODEL;
    return run_pipeline(video, model) ? 0 : 1;
}