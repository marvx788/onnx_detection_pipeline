/*
 * YOLO Detection — MJPEG Stream + Frame-by-Frame Kinematic Predictor
 * Optimized for Raspberry Pi 4 (OpenCV 4.x + OpenVINO C API)
 *
 * Principle:
 *   - Inference runs in the background at a variable rate (adaptive skip)
 *   - Once inference has produced >= 2 detections for an object, we compute
 *     velocity V = (pos_n - pos_{n-1}) / DeltaT
 *     where DeltaT = n_frames_elapsed * (1/DISPLAY_FPS)
 *   - Each display frame is BLANK: the prediction is recomputed from the
 *     last inference result: pos = pos_n + V * k * dt_display
 *     with k = number of frames since the last inference result
 *   - NO persistence of predictions between frames
 *   - The tracker is only used for inference->object matching (stable ID)
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
#include <atomic>

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

/* Set to 1 to enable profiling instrumentation */
#define ENABLE_PROFILING 1

/* ─── CONFIGURATION ──────────────────────────────────────────────────────── */
#define BASE_DIR       "/home/comvis/Documents/onnx_detection_pipeline"
static const char *DEFAULT_VIDEO = BASE_DIR "/test-video.mp4";
static const char *DEFAULT_MODEL = BASE_DIR "/yolo11n_256.onnx";

static const int    INPUT_SIZE      = 256;

/*
 * DISPLAY_SCALE is fixed at 1.0: the display frame is no longer resized
 * (see run_pipeline). It still exists because it scales detection boxes
 * in ai_thread_fn — at 1.0f this is a no-op, keeping boxes aligned with
 * the unscaled source frame sent to the MJPEG stream.
 */
static const float  DISPLAY_SCALE  = 1.0f;
static const int    JPEG_QUALITY   = 70;
static const int    HTTP_PORT      = 8080;

static const float  CONF_KEEP      = 0.25f;
static const float  NMS_THRESHOLD  = 0.45f;

static const double DISPLAY_FPS    = 25.0;   // FIXED DISPLAY FPS
static const double DT_DISPLAY      = 1.0 / DISPLAY_FPS; // duration of one display frame

/* Tracking / kinematic prediction */
static const float  MC_IOU_MATCH       = 0.25f;
static const int    TRACKER_MAX_AGE    = 3;    // missed inference frames before removal

/* Exponential smoothing for box sizes (avoids drastic jumps) */
static const float  SIZE_EMA_ALPHA     = 0.3f;

static const int    MAX_OBJECTS    = 64;
static const int    MAX_CLASSES    = 80;

/* Adaptive inference skip */
static const double TARGET_AI_MS   = 120.0;
static const int    MAX_SKIP       = 8;
static const double AI_EMA_ALPHA   = 0.2;

/* ─── COCO CLASSES ───────────────────────────────────────────────────────── */
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

/* ─── TIME UTILITIES ─────────────────────────────────────────────────────── */
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

/*
 * Adaptive skip, lock-free.
 * update() is only called by the inference thread (sole writer of
 * ai_ms_ema and current_skip). should_process() is only called by the
 * main thread (sole writer of frame_counter). Since no field ever has
 * two concurrent writers, relaxed atomics are sufficient here.
 */
struct AdaptiveSkip {
    std::atomic<double> ai_ms_ema{100.0};
    std::atomic<int>    current_skip{1};
    std::atomic<int>    frame_counter{0};

    void update(double ai_ms) {
        double old_ema = ai_ms_ema.load(std::memory_order_relaxed);
        double new_ema = AI_EMA_ALPHA * ai_ms + (1.0 - AI_EMA_ALPHA) * old_ema;
        ai_ms_ema.store(new_ema, std::memory_order_relaxed);

        double ideal_skip = new_ema * DISPLAY_FPS / 1000.0;
        int skip = std::max(1, std::min(MAX_SKIP, (int)std::ceil(ideal_skip)));
        current_skip.store(skip, std::memory_order_relaxed);
    }

    bool should_process() {
        int skip = current_skip.load(std::memory_order_relaxed);
        int fc = frame_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        if (fc >= skip) {
            frame_counter.store(0, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    int get_skip() { return current_skip.load(std::memory_order_relaxed); }
};

/* ─── DETECTION TYPES ────────────────────────────────────────────────────── */
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

/*
 * FRAME-BY-FRAME KINEMATIC PREDICTOR — scale-factor corrected.
 * Velocity is only recomputed on a new inference result via push();
 * the display frame just does a few multiply/adds in predict().
 */
struct KinState {
    float cx_cur, cy_cur, w_cur, h_cur;
    float conf_cur;
    int   frame_idx_cur;

    float cx_prev, cy_prev, w_prev, h_prev;
    int   frame_idx_prev;

    float L_cur  = 0.0f;
    float L_prev = 0.0f;

    bool has_two = false;
    bool initialized = false;

    float w_smooth = 0, h_smooth = 0;
    bool  smooth_init = false;

    void push(const Box &b, float conf, int frame_idx) {
        if (!smooth_init) {
            w_smooth = b.w;
            h_smooth = b.h;
            smooth_init = true;
        } else {
            w_smooth = SIZE_EMA_ALPHA * b.w + (1.0f - SIZE_EMA_ALPHA) * w_smooth;
            h_smooth = SIZE_EMA_ALPHA * b.h + (1.0f - SIZE_EMA_ALPHA) * h_smooth;
        }

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
            cx_prev = cx_cur; cy_prev = cy_cur;
            w_prev  = w_cur;  h_prev  = h_cur;
            frame_idx_prev = frame_idx_cur;
            L_prev = L_cur;

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

    bool predict(int frame_idx_now, Box &out_box, float &out_conf) const {
        if (!has_two)
            return false;

        int delta_frames = frame_idx_cur - frame_idx_prev;
        if (delta_frames < 1)
            delta_frames = 1;

        int k = frame_idx_now - frame_idx_cur;
        if (k < 0)
            k = 0;

        if (k > delta_frames * 2)
            return false;

        float scale_factor = 1.0f;
        if (L_prev > 1e-6f && L_cur > 1e-6f) {
            scale_factor = L_cur / L_prev;
            scale_factor = std::max(0.5f, std::min(2.0f, scale_factor));
        }

        float DeltaT = delta_frames / (float)DISPLAY_FPS;
        float dt = k / (float)DISPLAY_FPS;

        float V_cx = (cx_cur - cx_prev) / DeltaT;
        float V_cy = (cy_cur - cy_prev) / DeltaT;
        float V_w  = (w_cur  - w_prev)  / DeltaT;
        float V_h  = (h_cur  - h_prev)  / DeltaT;

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

/* ─── INTERNAL TRACKER (inference -> object matching) ────────────────────── */
struct TrackedObject {
    int         id;
    int         cls_id;
    cv::Scalar  color;
    int         consecutive_missed;
    int         age;
    KinState    kin;
    int         last_ai_frame_idx;
    Box         last_box;
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

            if (matched[oi] || t.objects[oi].cls_id != d.cls_id)
                continue;

            Box ref;
            float tmp_conf;

            if (!t.objects[oi].kin.predict(frame_idx, ref, tmp_conf)) {
                ref = t.objects[oi].last_box;
            }

            float iou = iou_boxes(d.box, ref);
            float cdist = center_dist_norm(d.box, ref);

            float score = 2.f * iou + 2.f * std::max(0.f, 1.f - cdist);

            if (score > best_score && (iou > 0.05f || cdist < 1.2f)) {
                best_score = score;
                best_obj = oi;
            }
        }

        if (best_obj >= 0) {
            TrackedObject &obj = t.objects[best_obj];
            obj.last_box = d.box;
            obj.last_conf = d.conf;
            obj.last_ai_frame_idx = frame_idx;
            obj.consecutive_missed = 0;
            obj.age++;
            obj.kin.push(d.box, d.conf, frame_idx);
            matched[best_obj] = true;

        } else if ((int)t.objects.size() < MAX_OBJECTS) {
            TrackedObject obj{};
            obj.id = ++t.next_id;
            obj.cls_id = d.cls_id;
            obj.color = CLASS_COLORS[d.cls_id % 80];
            obj.last_box = d.box;
            obj.last_conf = d.conf;
            obj.last_ai_frame_idx = frame_idx;
            obj.age = 1;
            obj.consecutive_missed = 0;
            obj.kin.push(d.box, d.conf, frame_idx);
            t.objects.push_back(obj);
            matched.push_back(false);
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
 * No NMS on kinematic predictions: each tracked object produces at most
 * one predicted box (one object = one TrackedObject = one kin.predict()
 * call), so there are no duplicates to filter here.
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

    pthread_mutex_unlock(&t.lock);
    return result;
}

/* ─── OPENVINO INFERENCE ─────────────────────────────────────────────────── */
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

    /* Limit the number of threads the CPU plugin uses for inference. On a
     * Pi4 (4 cores), reserve 2 for inference and leave 2 for the
     * capture/encode/display threads. CPU-only, no effect on GPU. */
    if (strcmp(device, "CPU") == 0) {
        ov_status_e st = ov_core_set_property(
            m.core, "CPU",
            "INFERENCE_NUM_THREADS", "2",
            nullptr);  // required terminator
        if (st != OK) {
            fprintf(stderr,
                "[OV] Warning: could not limit INFERENCE_NUM_THREADS "
                "(code %d) — inference will use all cores by default.\n",
                (int)st);
        }
    }

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

/*
 * ov_infer takes an image already resized to INPUT_SIZE x INPUT_SIZE
 * (done once by the producer, see the main loop).
 *
 * orig_w / orig_h are the ORIGINAL frame dimensions (before the producer's
 * resize), needed to rescale detected boxes back to display coordinates.
 *
 * rgb/tensor are thread_local: since this function is only called from the
 * inference thread, they are allocated once and reused across calls.
 */
static std::vector<Detection> ov_infer(OVModel &m, const cv::Mat &small_bgr,
                                        int orig_w, int orig_h, double* out_ms) {
    thread_local cv::Mat rgb;
    thread_local std::vector<float> tensor(3 * INPUT_SIZE * INPUT_SIZE);

    cv::cvtColor(small_bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);

    /* Row-by-row pointer access instead of .at<>() (faster) */
    for (int r = 0; r < INPUT_SIZE; r++) {
        const cv::Vec3f *row = rgb.ptr<cv::Vec3f>(r);
        float *pr = tensor.data() + 0*INPUT_SIZE*INPUT_SIZE + r*INPUT_SIZE;
        float *pg = tensor.data() + 1*INPUT_SIZE*INPUT_SIZE + r*INPUT_SIZE;
        float *pb = tensor.data() + 2*INPUT_SIZE*INPUT_SIZE + r*INPUT_SIZE;
        for (int col = 0; col < INPUT_SIZE; col++) {
            const cv::Vec3f &px = row[col];
            pr[col] = px[0];
            pg[col] = px[1];
            pb[col] = px[2];
        }
    }

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

    float xs = (float)orig_w / INPUT_SIZE;
    float ys = (float)orig_h / INPUT_SIZE;

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

static void draw_detections(cv::Mat &frame, const std::vector<Detection> &dets) {
    for (const auto &d : dets) {
        int x = (int)d.box.x, y = (int)d.box.y;
        int w = (int)d.box.w, h = (int)d.box.h;
        cv::Scalar color = CLASS_COLORS[d.cls_id % 80];
        cv::rectangle(frame, cv::Rect(x, y, w, h), color, 2, cv::LINE_8);

        char label[80];
        snprintf(label, sizeof(label), "%s: %.0f%%",
                 COCO_CLASSES[d.cls_id % 80], d.conf * 100.f);

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN,
                                      1.1, 1, &baseline);
        int ly = std::max(y - 4, ts.height + 4);
        cv::rectangle(frame,
                      cv::Point(x, ly - ts.height - baseline - 4),
                      cv::Point(x + ts.width + 4, ly),
                      color, cv::FILLED);
        cv::putText(frame, label, cv::Point(x+2, ly-baseline-2),
                    cv::FONT_HERSHEY_PLAIN, 1.1,
                    cv::Scalar(255,255,255), 1, cv::LINE_8);
    }
}

/* ─── SHARED JPEG BUFFER ─────────────────────────────────────────────────── */
static pthread_mutex_t  g_jpeg_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   g_jpeg_cond = PTHREAD_COND_INITIALIZER;
static std::vector<uint8_t> g_jpeg_buf;
static std::atomic<int> g_jpeg_seq{0};

/* ─── MJPEG SERVER ────────────────────────────────────────────────────────── */
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
            int cur_seq = g_jpeg_seq.load(std::memory_order_relaxed);
            if (cur_seq == last_sent) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_sec += 1;
                pthread_cond_timedwait(&g_jpeg_cond, &g_jpeg_lock, &ts);
                cur_seq = g_jpeg_seq.load(std::memory_order_relaxed);
            }
            if (!g_jpeg_buf.empty() && cur_seq != last_sent) {
                jpg_copy   = g_jpeg_buf;
                last_sent  = cur_seq;
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

/* ─── INFERENCE THREAD ───────────────────────────────────────────────────── */
/*
 * AIItem carries an image already resized to INPUT_SIZE x INPUT_SIZE
 * (small_bgr), moved from a fresh local Mat in the main thread — no
 * copy/clone needed.
 */
struct AIItem {
    cv::Mat small_bgr;
    int     frame_idx;
    int     orig_w;
    int     orig_h;
};

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
        std::vector<Detection> dets = ov_infer(*a->ov, item.small_bgr,
                                                item.orig_w, item.orig_h,
                                                &infer_ms);

        a->skip->update(infer_ms);

        /* No-op while DISPLAY_SCALE == 1.0f; kept for when display
         * resizing is reintroduced. */
        if (DISPLAY_SCALE != 1.0f)
            for (auto &d : dets) {
                d.box.x *= DISPLAY_SCALE; d.box.y *= DISPLAY_SCALE;
                d.box.w *= DISPLAY_SCALE; d.box.h *= DISPLAY_SCALE;
            }

        tracker_update(*a->tracker, dets, item.frame_idx);
    }
    return NULL;
}

/* ─── JPEG ENCODE THREAD ─────────────────────────────────────────────────── */
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

        if (frame.empty()) {
            usleep(5000);
            continue;
        }

        try {
            cv::imencode(".jpg", frame, jpeg_buf, params);
        }
        catch (...) {
            continue;
        }

        pthread_mutex_lock(&g_jpeg_lock);
        g_jpeg_buf = jpeg_buf;
        g_jpeg_seq.fetch_add(1, std::memory_order_relaxed);
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

#if ENABLE_PROFILING
/* Simple accumulators, printed every ~5s to stderr so as not to pollute
 * stdout (progress bar). */
struct ProfAccum {
    double capture = 0, resize_enqueue = 0, predict = 0, draw = 0, enqueue_jpeg = 0;
    int    n = 0;

    void add(double a, double b, double c, double d, double e) {
        capture += a; resize_enqueue += b; predict += c; draw += d; enqueue_jpeg += e;
        n++;
    }

    void maybe_print(int period) {
        if (n < period) return;
        fprintf(stderr,
            "[profile] capture=%.2fms  resize+ai_enqueue=%.2fms  predict=%.2fms  "
            "draw=%.2fms  enqueue_jpeg=%.2fms  (avg over %d frames)\n",
            1000.0*capture/n, 1000.0*resize_enqueue/n, 1000.0*predict/n,
            1000.0*draw/n, 1000.0*enqueue_jpeg/n, n);
        capture = resize_enqueue = predict = draw = enqueue_jpeg = 0;
        n = 0;
    }
};
#endif

/* ─── MAIN PIPELINE ───────────────────────────────────────────────────────── */
static bool run_pipeline(const char *video_path, const char *model_path) {
    init_colors();
    signal(SIGINT, handle_sigint);

    char pipeline[1024];
    bool is_camera = strncmp(video_path, "/dev/video30", 10) == 0;

    if (is_camera) {
        snprintf(pipeline, sizeof(pipeline),
        "v4l2src device=%s ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "videoscale ! "
        "video/x-raw,width=640,height=360 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink sync=false drop=true max-buffers=1",
        video_path);
    } else {
        snprintf(pipeline, sizeof(pipeline),
        "filesrc location=%s ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "decodebin ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "videoscale ! "
        "video/x-raw,width=640,height=360 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink sync=false drop=true",
        video_path);
    }

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        fprintf(stderr, "[ERROR] Could not open: %s\n", video_path);
        return false;
    }
    int    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps    = cap.get(cv::CAP_PROP_FPS);
    int    total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    int    disp_w = width;
    int    disp_h = height;
    double source_duration = (total > 0 && fps > 0) ? total / fps : 0.0;

    printf("================================================\n");
    printf("  YOLO — MJPEG Stream (Kinematic Predictor) — OPTIMIZED\n");
    printf("================================================\n");
    printf("  Input   : %s\n", video_path);
    printf("  Source  : %dx%d @ %.1ffps (%d frames, %.1fs)\n",
           width, height, fps, total, source_duration);
    printf("  Display : %dx%d @ %.1ffps (FIXED, no resize)\n", disp_w, disp_h, DISPLAY_FPS);
    printf("  Predictor : pure frame-by-frame kinematic\n");
    printf("  Loading ONNX model...\n");

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

    printf("  MJPEG server started on port %d\n", HTTP_PORT);
    printf("  Waiting for 2 inference results before drawing boxes...\n\n");

    int    displayed = 0, ai_processed = 0;
    double t_program_start = perf_counter();
    int    frame_idx = 0;
    double last_disp_t = 0.0, disp_fps = 0.0;
    double disp_fps_acc = 0.0; int disp_fps_n = 0;
    double t_real_start = perf_counter();

#if ENABLE_PROFILING
    ProfAccum prof;
    const int PROF_PERIOD = (int)(DISPLAY_FPS * 5); // ~5s
#endif

    while (!g_stop) {
#if ENABLE_PROFILING
        double tp0 = perf_counter();
#endif
        cv::Mat raw;
        cap >> raw;

        double actual_fps = disp_fps > 1.0 ? disp_fps : DISPLAY_FPS;

        if (actual_fps < DISPLAY_FPS * 0.95) {
            int skip_frames = (int)(DISPLAY_FPS / actual_fps) - 1;
            for (int i = 0; i < skip_frames; i++) {
                cv::Mat dummy;
                if (!cap.read(dummy)) break;
                frame_idx++;
            }
        }

        /*
         * BUGFIX: an empty frame from a video FILE means "end of file"
         * (break is correct). From a CAMERA, an empty frame is a
         * TRANSIENT GStreamer/v4l2 read error (e.g. "unable to query
         * pipeline state" / "v4l2src0 reported: Failed to allocate a
         * buffer") — not end of stream. In camera mode we now attempt
         * to RECONNECT the pipeline instead of exiting the program.
         */
        if (raw.empty()) {
            if (is_camera) {
                fprintf(stderr,
                    "\n[WARN] Camera read failed (empty frame) — "
                    "attempting to reconnect the v4l2 pipeline...\n");
                cap.release();

                const int    MAX_RETRY      = 50;   // ~10s total
                const double RETRY_DELAY_S  = 0.2;
                bool reopened = false;
                int  retry = 0;

                while (!g_stop && retry < MAX_RETRY) {
                    sleep_sec(RETRY_DELAY_S);
                    cap.open(pipeline, cv::CAP_GSTREAMER);
                    if (cap.isOpened()) { reopened = true; break; }
                    retry++;
                }

                if (!reopened) {
                    fprintf(stderr,
                        "[ERROR] Camera reconnection failed after %d "
                        "attempts — stopping pipeline.\n", MAX_RETRY);
                    break;
                }

                fprintf(stderr,
                    "[INFO] Camera pipeline reconnected after %d attempt(s).\n",
                    retry);
                continue;
            } else {
                /* Video file: normal end of stream. */
                break;
            }
        }

        frame_idx++; // display frame number (1, 2, 3, ...)

#if ENABLE_PROFILING
        double tp1 = perf_counter();
#endif

        bool do_ai = skip_controller.should_process();
        if (do_ai) {
            cv::Mat small;
            cv::resize(raw, small, cv::Size(INPUT_SIZE, INPUT_SIZE));

            pthread_mutex_lock(&ai_q.lock);
            ai_q.slot = { std::move(small), frame_idx, raw.cols, raw.rows };
            ai_q.has_item = true;
            pthread_cond_signal(&ai_q.cond_consumer);
            pthread_mutex_unlock(&ai_q.lock);
            ai_processed++;
        }

#if ENABLE_PROFILING
        double tp2 = perf_counter();
#endif

        cv::Mat disp = raw.clone();

        std::vector<Detection> pred_dets =
            tracker_predict_kinematic(tracker, frame_idx);

#if ENABLE_PROFILING
        double tp3 = perf_counter();
#endif

        draw_detections(disp, pred_dets);

#if ENABLE_PROFILING
        double tp4 = perf_counter();
#endif

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
                    0.6, cv::Scalar(0,0,0), 3, cv::LINE_8);
        cv::putText(disp, osd, cv::Point(10,28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0,220,80), 2, cv::LINE_8);

        /* ─── SEND TO JPEG ENCODER ─── */
        pthread_mutex_lock(&enc_slot.lock);
        enc_slot.frame = disp;
        enc_slot.has_frame = true;
        pthread_mutex_unlock(&enc_slot.lock);

        displayed++;

#if ENABLE_PROFILING
        double tp5 = perf_counter();
        prof.add(tp1-tp0, tp2-tp1, tp3-tp2, tp4-tp3, tp5-tp4);
        prof.maybe_print(PROF_PERIOD);
#endif

        /* ─── PROGRESS BAR ─── */
        float pct = (total > 0) ? (100.f * frame_idx / total) : 0.f;
        int bar_n = (int)(25 * pct / 100);
        char bar[27] = {};
        for (int i=0; i<25; i++) bar[i] = i < bar_n ? '#' : '-';
        printf("\r  [%s] %5.1f%%  F:%d  skip:%d  Disp:%5.1ffps  AI:%5.1ffps  Obj:%d(%dP)",
               bar, (double)pct, frame_idx, current_skip,
               disp_fps, ai_fps, n_tracked, n_predictable);
        fflush(stdout);

        /* ─── FIXED-FPS SYNC ─── */
        double next_display_time = t_program_start + displayed / DISPLAY_FPS;
        double sleep_t = next_display_time - perf_counter();

        if (sleep_t > 0.0) {
            sleep_sec(sleep_t);
        } else if (sleep_t < -0.1) {
            t_program_start += 0.05;
        }
    }

    printf("\n\n  Finishing up...\n");
    g_stop = 1;
    ai_arg.running = enc_arg.running = false;
    pthread_cond_broadcast(&ai_q.cond_consumer);
    pthread_cond_broadcast(&g_jpeg_cond);

    pthread_join(t_ai,  NULL);
    pthread_join(t_enc, NULL);

    cap.release();
    ov_free(ov);

    double elapsed = perf_counter() - t_real_start;
    printf("  Frames read      : %d\n", frame_idx);
    printf("  Frames displayed : %d\n", displayed);
    printf("  Frames inferred  : %d\n", ai_processed);
    printf("  Elapsed time     : %.1fs\n", elapsed);
    double real_source_duration = (frame_idx > 0 && fps > 0) ? frame_idx / fps : 0.0;

    printf("  Source duration  : %.1fs\n", real_source_duration);
    printf("  Avg display FPS  : %.1f\n", displayed / elapsed);
    printf("  Done!\n");
    return true;
}

int main(int argc, char *argv[]) {
    const char *video = argc > 1 ? argv[1] : DEFAULT_VIDEO;
    const char *model = argc > 2 ? argv[2] : DEFAULT_MODEL;
    return run_pipeline(video, model) ? 0 : 1;
}