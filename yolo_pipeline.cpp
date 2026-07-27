/*
 * YOLO Detection — MJPEG Stream + Forward Temporal Prediction
 * Traduction C++11 (OpenCV 4.x + OpenVINO C API)
 *
 * OpenCV 4.x a supprimé l'API C pure (IplImage, cvResize…).
 * On utilise cv::Mat / l'API C++ moderne tout en gardant
 * le reste de la logique identique au Python v6.1.
 */

/* _GNU_SOURCE défini par g++ automatiquement */
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
static const float  DISPLAY_SCALE  = 0.75f;
static const int    JPEG_QUALITY   = 70;
static const int    HTTP_PORT      = 8080;

static const float  CONF_KEEP      = 0.25f;
static const float  NMS_THRESHOLD  = 0.45f;

static const char  *RECORD_OUTPUT  = "output_compensation.mp4";
static const double RECORD_FPS     = 12.5;

static const bool   MC_ENABLED         = true;
static const double MC_MAX_PREDICT_MS  = 500.0;
static const float  MC_DECAY_CONF      = 0.02f;
static const float  MC_MIN_CONF        = 0.10f;
static const float  MC_IOU_MATCH       = 0.30f;

static const int    MAX_OBJECTS    = 64;
static const int    MAX_DETECTIONS = 64;
static const int    MAX_CLASSES    = 80;

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

/* ─── TYPES DÉTECTION / TRACKING ────────────────────────────────────────── */
struct Box { float x, y, w, h; };

struct Detection {
    Box   box;
    float conf;
    int   cls_id;
};

struct TrackedObject {
    int    id, cls_id;
    Box    box, box_ia;
    float  conf, conf_ia;
    double t_capture;
    float  vx_ps, vy_ps, vw_ps, vh_ps;
    double last_update;
    int    frames_since_ia, age;
    cv::Scalar color;
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

/* ─── TRACKER ────────────────────────────────────────────────────────────── */
struct Tracker {
    std::vector<TrackedObject> objects;
    int    max_age;
    float  iou_threshold;
    pthread_mutex_t lock;
    int    next_id = 0;
    int    frame_counter = 0;

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

static void update_velocity(TrackedObject &obj, const Box &nb,
                             double t_cap, float alpha) {
    double dt = t_cap - obj.t_capture;
    if (dt < 1e-4) dt = 0.04;
    float vx = (nb.x - obj.box_ia.x) / (float)dt;
    float vy = (nb.y - obj.box_ia.y) / (float)dt;
    float vw = (nb.w - obj.box_ia.w) / (float)dt;
    float vh = (nb.h - obj.box_ia.h) / (float)dt;
    obj.vx_ps = alpha*vx + (1-alpha)*obj.vx_ps;
    obj.vy_ps = alpha*vy + (1-alpha)*obj.vy_ps;
    obj.vw_ps = alpha*vw + (1-alpha)*obj.vw_ps;
    obj.vh_ps = alpha*vh + (1-alpha)*obj.vh_ps;
}

static void tracker_update(Tracker &t,
                            const std::vector<Detection> &dets,
                            double t_capture) {
    pthread_mutex_lock(&t.lock);

    std::vector<bool> matched(t.objects.size(), false);

    for (const auto &d : dets) {
        int   best_obj   = -1;
        float best_score = -1.f;

        for (size_t oi = 0; oi < t.objects.size(); oi++) {
            if (matched[oi] || t.objects[oi].cls_id != d.cls_id) continue;
            const Box &ref = t.objects[oi].box_ia;
            float iou   = iou_boxes(d.box, ref);
            float cdist = center_dist_norm(d.box, ref);
            float score = iou * 2.f + std::max(0.f, 1.f - cdist);
            if (score > best_score && iou >= t.iou_threshold) {
                best_score = score; best_obj = (int)oi;
            }
        }

        if (best_obj >= 0) {
            TrackedObject &obj = t.objects[best_obj];
            update_velocity(obj, d.box, t_capture, 0.6f);
            obj.box = obj.box_ia   = d.box;
            obj.conf = obj.conf_ia = d.conf;
            obj.t_capture  = t_capture;
            obj.last_update = t_capture;
            obj.frames_since_ia = 0;
            obj.age++;
            matched[best_obj] = true;
        } else if ((int)t.objects.size() < MAX_OBJECTS) {
            TrackedObject obj{};
            obj.id = ++t.next_id;
            obj.cls_id = d.cls_id;
            obj.box = obj.box_ia = d.box;
            obj.conf = obj.conf_ia = d.conf;
            obj.t_capture = obj.last_update = t_capture;
            obj.age = 1;
            obj.color = CLASS_COLORS[d.cls_id % 80];
            t.objects.push_back(obj);
            matched.push_back(false); /* pour les nouvellement ajoutés */
        }
    }

    /* Incrémenter frames_since_ia pour les non mis-à-jour */
    for (size_t oi = 0; oi < t.objects.size(); oi++)
        if (!matched[oi] && t.objects[oi].frames_since_ia > 0)
            t.objects[oi].frames_since_ia++;

    /* Purger */
    t.objects.erase(
        std::remove_if(t.objects.begin(), t.objects.end(),
            [&](const TrackedObject &o){
                return o.frames_since_ia >= t.max_age || o.conf <= MC_MIN_CONF;
            }),
        t.objects.end());

    pthread_mutex_unlock(&t.lock);
}

static std::vector<Detection> tracker_predict(Tracker &t, double t_now) {
    pthread_mutex_lock(&t.lock);
    t.frame_counter++;
    std::vector<Detection> result;

    for (auto it = t.objects.begin(); it != t.objects.end(); ) {
        double dt = t_now - it->t_capture;
        if (dt > MC_MAX_PREDICT_MS / 1000.0) {
            it = t.objects.erase(it); continue;
        }
        it->frames_since_ia++;
        float x = it->box_ia.x + it->vx_ps * (float)dt;
        float y = it->box_ia.y + it->vy_ps * (float)dt;
        float w = std::max(1.f, it->box_ia.w + it->vw_ps * (float)dt);
        float h = std::max(1.f, it->box_ia.h + it->vh_ps * (float)dt);
        it->box = {x, y, w, h};
        float conf = std::max(MC_MIN_CONF, it->conf_ia - MC_DECAY_CONF*(float)dt*25.f);
        it->conf = conf;
        result.push_back({{x, y, w, h}, conf, it->cls_id});
        ++it;
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
    /* Choisir GPU si disponible */
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

static std::vector<Detection> ov_infer(OVModel &m, const cv::Mat &frame) {
    int fw = frame.cols, fh = frame.rows;

    /* Resize + BGR→RGB + float32 normalisé, layout NCHW */
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
    m.count++; m.sum_ms += ms;
    pthread_mutex_unlock(&m.lock);

    /* Récupérer la sortie : shape [1, 84, n_anchors] */
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
        /* Layout [row * n_anchors + col] */
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
struct AIItem { cv::Mat frame; double t_capture; };

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
    AIQueue  *ai_q;
    OVModel  *ov;
    Tracker  *tracker;
    bool      running;
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

        std::vector<Detection> dets = ov_infer(*a->ov, item.frame);

        /* Mise à l'échelle display avant le tracker */
        if (DISPLAY_SCALE != 1.0f)
            for (auto &d : dets) {
                d.box.x *= DISPLAY_SCALE; d.box.y *= DISPLAY_SCALE;
                d.box.w *= DISPLAY_SCALE; d.box.h *= DISPLAY_SCALE;
            }

        tracker_update(*a->tracker, dets, item.t_capture);
    }
    return NULL;
}

/* ─── THREAD ENCODAGE JPEG ───────────────────────────────────────────────── */
struct EncodeSlot {
    cv::Mat         frame;
    bool            has_frame = false;
    pthread_mutex_t lock;
    EncodeSlot() { pthread_mutex_init(&lock, NULL); }
    ~EncodeSlot() { pthread_mutex_destroy(&lock); }
};

struct EncodeThreadArg {
    EncodeSlot      *slot;
    cv::VideoWriter *writer;
    pthread_mutex_t  writer_lock;
    bool             running;
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

        cv::imencode(".jpg", frame, jpeg_buf, params);

        pthread_mutex_lock(&g_jpeg_lock);
        g_jpeg_buf = jpeg_buf;
        g_jpeg_seq++;
        pthread_cond_broadcast(&g_jpeg_cond);
        pthread_mutex_unlock(&g_jpeg_lock);

        if (a->writer && a->writer->isOpened()) {
            pthread_mutex_lock(&a->writer_lock);
            a->writer->write(frame);
            pthread_mutex_unlock(&a->writer_lock);
        }
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

    cv::VideoCapture cap(video_path);
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

    printf("================================================\n");
    printf("  YOLO — MJPEG Stream (Forward Temporal Pred.)\n");
    printf("================================================\n");
    printf("  Input   : %s\n", video_path);
    printf("  Source  : %dx%d @ %.1ffps (%d frames)\n", width, height, fps, total);
    printf("  Display : %dx%d (scale %.2f)\n", disp_w, disp_h, DISPLAY_SCALE);
    printf("  Chargement modèle ONNX...\n");

    OVModel ov;
    if (!ov_init(ov, model_path)) return false;

    Tracker tracker(5, MC_IOU_MATCH);

    cv::VideoWriter writer(RECORD_OUTPUT,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           RECORD_FPS, cv::Size(disp_w, disp_h));

    AIQueue    ai_q;
    EncodeSlot enc_slot;

    AIThreadArg ai_arg { &ai_q, &ov, &tracker, true };
    pthread_t t_ai;
    pthread_create(&t_ai, NULL, ai_thread_fn, &ai_arg);

    EncodeThreadArg enc_arg;
    enc_arg.slot    = &enc_slot;
    enc_arg.writer  = &writer;
    enc_arg.running = true;
    pthread_mutex_init(&enc_arg.writer_lock, NULL);
    pthread_t t_enc;
    pthread_create(&t_enc, NULL, encode_thread_fn, &enc_arg);

    int *port_ptr = (int*)malloc(sizeof(int));
    *port_ptr = HTTP_PORT;
    pthread_t t_http;
    pthread_create(&t_http, NULL, http_server_thread, port_ptr);
    pthread_detach(t_http);

    printf("  Serveur MJPEG démarré sur le port %d\n", HTTP_PORT);
    printf("  Démarrage...\n\n");

    int    displayed = 0, ai_skipped = 0;
    double t_start   = perf_counter();
    int    frame_idx = 0;
    double last_disp_t = 0.0, disp_fps = 0.0;
    double disp_fps_acc = 0.0; int disp_fps_n = 0;

    while (!g_stop) {
        cv::Mat raw;
        cap >> raw;
        double t_capture = perf_counter();
        if (raw.empty()) break;
        cv::Mat dummy; cap >> dummy; /* skip frame */

        /* Envoyer au thread IA */
        pthread_mutex_lock(&ai_q.lock);
        if (ai_q.has_item) { ai_skipped++; }
        ai_q.slot = { raw.clone(), t_capture };
        ai_q.has_item = true;
        pthread_cond_signal(&ai_q.cond_consumer);
        pthread_mutex_unlock(&ai_q.lock);

        /* Frame d'affichage */
        cv::Mat disp;
        cv::resize(raw, disp, cv::Size(disp_w, disp_h));

        /* Forward temporal prediction */
        std::vector<Detection> pred_dets;
        if (MC_ENABLED) {
            double t_now = perf_counter();
            pred_dets = tracker_predict(tracker, t_now);
        }
        draw_detections(disp, pred_dets);

        /* OSD */
        double now = perf_counter();
        if (last_disp_t > 0.0) {
            double dt = now - last_disp_t;
            if (dt > 1e-9) { disp_fps_acc += 1.0/dt; disp_fps_n++; }
        }
        last_disp_t = now;
        if (disp_fps_n > 0) disp_fps = disp_fps_acc / disp_fps_n;

        double ai_fps = 0.0;
        pthread_mutex_lock(&ov.lock);
        if (ov.count > 0) ai_fps = ov.count / (ov.sum_ms / 1000.0);
        pthread_mutex_unlock(&ov.lock);

        int n_tracked = 0, n_predicted_mc = 0;
        pthread_mutex_lock(&tracker.lock);
        n_tracked = (int)tracker.objects.size();
        for (auto &o : tracker.objects)
            if ((now - o.t_capture) > 0.02) n_predicted_mc++;
        pthread_mutex_unlock(&tracker.lock);

        char osd[256];
        snprintf(osd, sizeof(osd),
                 "Disp:%.1ffps  AI:%.1ffps  F:%d  skip:%d  MC:%d(%dP)",
                 disp_fps, ai_fps, displayed, ai_skipped,
                 n_tracked, n_predicted_mc);
        cv::putText(disp, osd, cv::Point(10,28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0,0,0), 3, cv::LINE_AA);
        cv::putText(disp, osd, cv::Point(10,28), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0,220,80), 2, cv::LINE_AA);

        /* Envoyer à l'encodeur */
        pthread_mutex_lock(&enc_slot.lock);
        enc_slot.frame     = disp;
        enc_slot.has_frame = true;
        pthread_mutex_unlock(&enc_slot.lock);

        displayed++;

        float pct = (float)displayed / (float)(total/2 > 0 ? total/2 : 1) * 100.f;
        int bar_n = (int)(25 * pct / 100);
        char bar[27] = {};
        for (int i=0; i<25; i++) bar[i] = i < bar_n ? '#' : '-';
        printf("\r  [%s] %5.1f%%  Disp:%5.1ffps  AI:%5.1ffps  skip:%3d  MC:%d(%dP)",
               bar, (double)pct, disp_fps, ai_fps, ai_skipped,
               n_tracked, n_predicted_mc);
        fflush(stdout);

        /* Timing framerate */
        frame_idx++;
        double target  = t_start + frame_idx * (2.0 / fps);
        double sleep_t = target - perf_counter();
        if (sleep_t < -0.04) { t_start += 0.02; sleep_t = target - perf_counter(); }
        if (sleep_t > 0.0) sleep_sec(sleep_t);
    }

    printf("\n\n  Finalisation...\n");
    g_stop = 1;
    ai_arg.running = enc_arg.running = false;
    pthread_cond_broadcast(&ai_q.cond_consumer);
    pthread_cond_broadcast(&g_jpeg_cond);

    pthread_join(t_ai,  NULL);
    pthread_join(t_enc, NULL);

    cap.release();
    writer.release();
    ov_free(ov);
    pthread_mutex_destroy(&enc_arg.writer_lock);

    double elapsed = perf_counter() - t_start;
    printf("  Frames affichées : %d\n", displayed);
    printf("  Skip IA          : %d\n", ai_skipped);
    printf("  FPS affiché moy. : %.1f\n", displayed / elapsed);
    printf("  Terminé !\n");
    return true;
}

int main(int argc, char *argv[]) {
    const char *video = argc > 1 ? argv[1] : DEFAULT_VIDEO;
    const char *model = argc > 2 ? argv[2] : DEFAULT_MODEL;
    return run_pipeline(video, model) ? 0 : 1;
}