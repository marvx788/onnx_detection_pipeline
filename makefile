# ─────────────────────────────────────────────────────────────────
#  Makefile — YOLO MJPEG Pipeline (C++11 / OpenCV 4.x)
#  Chemins hardcodés pour Raspberry Pi + openvino pip 2024.6
# ─────────────────────────────────────────────────────────────────

CXX    = g++
TARGET = yolo_pipeline_9

CXXFLAGS = -O2 -march=native -std=c++11 -Wall -Wextra \
           -ffast-math -funroll-loops

# ── OpenCV ────────────────────────────────────────────────────
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || \
                          pkg-config --cflags opencv  2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null || \
                          pkg-config --libs   opencv  2>/dev/null)

# ── OpenVINO (pip install openvino 2024.6) ────────────────────
OV_INC = /home/comvis/.local/lib/python3.9/site-packages/openvino/include
OV_LIB = /home/comvis/.local/lib/python3.9/site-packages/openvino/libs

OV_CFLAGS = -I$(OV_INC)
OV_LIBS   = -L$(OV_LIB) -lopenvino_c -Wl,-rpath,$(OV_LIB)

# ── Threads + math ────────────────────────────────────────────
EXTRA_LIBS = -lpthread -lm

# ── Assemblage ───────────────────────────────────────────────
ALL_CXXFLAGS = $(CXXFLAGS) $(OPENCV_CFLAGS) $(OV_CFLAGS)
ALL_LIBS     = $(OPENCV_LIBS) $(OV_LIBS) $(EXTRA_LIBS)

# ─────────────────────────────────────────────────────────────
.PHONY: all clean run

all: $(TARGET)
	@echo "✓  Build réussi : ./$(TARGET)"

$(TARGET): yolo_pipeline_9.cpp
	$(CXX) $(ALL_CXXFLAGS) $< -o $@ $(ALL_LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)