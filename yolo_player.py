#!/usr/bin/env python3
"""
YOLO Detection Player - GStreamer pipeline avec contrôles pause/play/seek
Usage: python3 yolo_player.py [video_path] [model_path]
"""

import sys
import os
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('Gtk', '3.0')

from gi.repository import Gst, GLib, Gtk, GObject

# === CONFIGURATION ===
DEFAULT_VIDEO  = "/home/marvin/Documents/onnx_detection/onnx_detection_gpu/build/test_video2.mp4"
DEFAULT_MODEL  = "/home/marvin/Documents/onnx_detection/onnx_detection_gpu/yolo11n.onnx"
PLUGIN_PATH    = "/home/marvin/Documents/onnx_detection/onnx_detection_gpu/gst-template/builddir/gst-plugin"
ONNX_LIB_PATH = "/opt/onnxruntime/lib"

# Appliquer les variables d'environnement
os.environ["GST_PLUGIN_PATH"] = PLUGIN_PATH
os.environ["LD_LIBRARY_PATH"] = ONNX_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ORT_DISABLE_CUDA"] = "1"

Gst.init(None)


class YoloPlayer(Gtk.Window):
    def __init__(self, video_path, model_path):
        super().__init__(title="YOLO Detection Player")
        self.video_path = video_path
        self.model_path = model_path
        self.pipeline = None
        self.duration = Gst.CLOCK_TIME_NONE
        self.is_playing = False
        self.seeking = False

        self._build_ui()
        self._build_pipeline()
        self.connect("destroy", self._on_destroy)
        self.show_all()

    # ------------------------------------------------------------------ UI ---

    def _build_ui(self):
        self.set_default_size(1000, 680)
        self.set_border_width(0)

        # CSS
        css = b"""
        window { background: #0d0d0d; }
        #toolbar {
            background: #1a1a1a;
            border-top: 1px solid #2a2a2a;
            padding: 10px 16px;
        }
        #btn-play {
            background: #00e5a0;
            color: #000;
            border: none;
            border-radius: 6px;
            padding: 8px 22px;
            font-weight: 700;
            font-size: 13px;
        }
        #btn-play:hover { background: #00ffb3; }
        #btn-stop {
            background: #2a2a2a;
            color: #aaa;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
        }
        #btn-stop:hover { background: #333; color: #fff; }
        #lbl-time {
            color: #666;
            font-family: monospace;
            font-size: 12px;
        }
        scale trough { background: #2a2a2a; border-radius: 4px; }
        scale highlight { background: #00e5a0; border-radius: 4px; }
        scale slider {
            background: #fff;
            border-radius: 50%;
            min-width: 14px;
            min-height: 14px;
        }
        #video-area { background: #000; }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(vbox)

        # Zone vidéo (dual view : original + YOLO)
        hbox_video = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        hbox_video.set_name("video-area")
        hbox_video.set_vexpand(True)

        self.video_widget_orig = Gtk.DrawingArea()
        self.video_widget_orig.set_hexpand(True)
        self.video_widget_orig.set_vexpand(True)

        self.video_widget_yolo = Gtk.DrawingArea()
        self.video_widget_yolo.set_hexpand(True)
        self.video_widget_yolo.set_vexpand(True)

        # Labels au-dessus de chaque fenêtre
        vbox_orig = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        lbl_orig = Gtk.Label(label="Original")
        lbl_orig.set_markup('<span foreground="#666" size="small">ORIGINAL</span>')
        lbl_orig.set_margin_top(6)
        vbox_orig.pack_start(lbl_orig, False, False, 0)
        vbox_orig.pack_start(self.video_widget_orig, True, True, 0)

        vbox_yolo = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        lbl_yolo = Gtk.Label(label="YOLO")
        lbl_yolo.set_markup('<span foreground="#00e5a0" size="small">YOLO DETECTION</span>')
        lbl_yolo.set_margin_top(6)
        vbox_yolo.pack_start(lbl_yolo, False, False, 0)
        vbox_yolo.pack_start(self.video_widget_yolo, True, True, 0)

        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        hbox_video.pack_start(vbox_orig, True, True, 0)
        hbox_video.pack_start(sep, False, False, 0)
        hbox_video.pack_start(vbox_yolo, True, True, 0)
        vbox.pack_start(hbox_video, True, True, 0)

        # Barre de progression
        self.seek_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL)
        self.seek_scale.set_range(0, 1)
        self.seek_scale.set_draw_value(False)
        self.seek_scale.set_margin_start(12)
        self.seek_scale.set_margin_end(12)
        self.seek_scale.set_margin_top(8)
        self.seek_scale.connect("button-press-event", self._on_seek_start)
        self.seek_scale.connect("button-release-event", self._on_seek_end)
        vbox.pack_start(self.seek_scale, False, False, 0)

        # Toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        toolbar.set_name("toolbar")

        self.btn_play = Gtk.Button(label="▶  PLAY")
        self.btn_play.set_name("btn-play")
        self.btn_play.connect("clicked", self._on_play_pause)

        btn_stop = Gtk.Button(label="■  Stop")
        btn_stop.set_name("btn-stop")
        btn_stop.connect("clicked", self._on_stop)

        self.lbl_time = Gtk.Label(label="0:00 / 0:00")
        self.lbl_time.set_name("lbl-time")

        # Vitesse de lecture
        lbl_speed = Gtk.Label()
        lbl_speed.set_markup('<span foreground="#555" size="small">Vitesse</span>')
        self.speed_combo = Gtk.ComboBoxText()
        for s in ["0.5x", "1x", "1.5x", "2x"]:
            self.speed_combo.append_text(s)
        self.speed_combo.set_active(1)
        self.speed_combo.connect("changed", self._on_speed_change)

        toolbar.pack_start(self.btn_play, False, False, 0)
        toolbar.pack_start(btn_stop, False, False, 0)
        toolbar.pack_start(Gtk.Separator(), False, False, 4)
        toolbar.pack_start(self.lbl_time, False, False, 0)
        toolbar.pack_end(self.speed_combo, False, False, 0)
        toolbar.pack_end(lbl_speed, False, False, 0)

        vbox.pack_start(toolbar, False, False, 0)

    # ---------------------------------------------------------- Pipeline ---

    def _build_pipeline(self):
        pipeline_str = f"""
            filesrc location={self.video_path} !
            decodebin name=dec
            dec. ! queue ! audioconvert ! fakesink
            dec. ! queue ! videoconvert ! videoscale ! videorate ! tee name=t
            t. ! queue max-size-buffers=10 ! videoconvert ! videoscale !
                 glimagesink name=sink_orig sync=true
            t. ! queue max-size-buffers=3 leaky=downstream ! videoconvert ! videoscale !
                 video/x-raw,format=RGB,width=640,height=640 !
                 onnxinference model-location={self.model_path} !
                 onnxpostprocess draw-results=true !
                 videoconvert !
                 glimagesink name=sink_yolo sync=true
        """
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Lier les sinks aux widgets GTK
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect("message", self._on_bus_message)
        bus.connect("sync-message::element", self._on_sync_message)

        # Timer pour mise à jour de la barre de progression
        GLib.timeout_add(500, self._update_progress)

    def _embed_video(self, message, widget):
        """Intègre la fenêtre GStreamer dans le widget GTK."""
        if message.get_structure().get_name() == "prepare-window-handle":
            message.src.set_property("force-aspect-ratio", True)
            win_id = widget.get_window().get_xid()
            message.src.set_window_handle(win_id)

    def _on_sync_message(self, bus, message):
        struct = message.get_structure()
        if not struct:
            return
        name = struct.get_name()
        if name == "prepare-window-handle":
            sink_name = message.src.get_name()
            if sink_name and "orig" in sink_name:
                GLib.idle_add(self._embed_video, message, self.video_widget_orig)
            else:
                GLib.idle_add(self._embed_video, message, self.video_widget_yolo)

    # ---------------------------------------------------------- Contrôles ---

    def _on_play_pause(self, widget):
        if not self.is_playing:
            self.pipeline.set_state(Gst.State.PLAYING)
            self.btn_play.set_label("⏸  PAUSE")
            self.is_playing = True
        else:
            self.pipeline.set_state(Gst.State.PAUSED)
            self.btn_play.set_label("▶  PLAY")
            self.is_playing = False

    def _on_stop(self, widget):
        self.pipeline.set_state(Gst.State.NULL)
        self.is_playing = False
        self.btn_play.set_label("▶  PLAY")
        self.seek_scale.set_value(0)
        self.lbl_time.set_text("0:00 / 0:00")

    def _on_seek_start(self, widget, event):
        self.seeking = True

    def _on_seek_end(self, widget, event):
        value = self.seek_scale.get_value()
        if self.duration != Gst.CLOCK_TIME_NONE:
            seek_pos = int(value * self.duration)
            self.pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                seek_pos
            )
        self.seeking = False

    def _on_speed_change(self, combo):
        speeds = {"0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}
        rate = speeds.get(combo.get_active_text(), 1.0)
        if self.pipeline and self.is_playing:
            ok, pos = self.pipeline.query_position(Gst.Format.TIME)
            if ok:
                self.pipeline.seek(
                    rate,
                    Gst.Format.TIME,
                    Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
                    Gst.SeekType.SET, pos,
                    Gst.SeekType.NONE, 0
                )

    def _update_progress(self):
        if self.pipeline and self.is_playing and not self.seeking:
            ok_pos, pos = self.pipeline.query_position(Gst.Format.TIME)
            ok_dur, dur = self.pipeline.query_duration(Gst.Format.TIME)
            if ok_dur:
                self.duration = dur
            if ok_pos and self.duration != Gst.CLOCK_TIME_NONE:
                frac = pos / self.duration
                self.seek_scale.set_value(frac)
                pos_s = pos // Gst.SECOND
                dur_s = self.duration // Gst.SECOND
                self.lbl_time.set_text(
                    f"{pos_s // 60}:{pos_s % 60:02d} / {dur_s // 60}:{dur_s % 60:02d}"
                )
        return True  # continuer le timer

    # ---------------------------------------------------------- Bus msgs ---

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_playing = False
            self.btn_play.set_label("▶  PLAY")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERREUR] {err.message}")
            print(f"[DEBUG]  {debug}")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_playing = False

    def _on_destroy(self, widget):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        Gtk.main_quit()


# ================================================================== MAIN ===

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    if not os.path.exists(video):
        print(f"[ERREUR] Vidéo introuvable : {video}")
        sys.exit(1)
    if not os.path.exists(model):
        print(f"[ERREUR] Modèle introuvable : {model}")
        sys.exit(1)

    print(f"Vidéo : {video}")
    print(f"Modèle : {model}")
    print(f"Plugins : {PLUGIN_PATH}")

    player = YoloPlayer(video, model)
    Gtk.main()