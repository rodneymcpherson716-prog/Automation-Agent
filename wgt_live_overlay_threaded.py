import os
import re
import time
import queue
import threading
import tkinter as tk

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT

REFRESH_MS = 250
SCAN_DELAY = 0.8

pyautogui.FAILSAFE = True


def get_wgt_window():
    matches = []

    try:
        for win in pyautogui.getAllWindows():
            try:
                title = (win.title or "").strip()

                if (
                    title.lower() == "wgt golf"
                    and int(win.width) >= 700
                    and int(win.height) >= 450
                ):
                    matches.append(win)
            except Exception:
                pass
    except Exception:
        pass

    if not matches:
        return None

    return max(
        matches,
        key=lambda w: int(w.width) * int(w.height),
    )


def capture_wgt(win, sct):
    monitor = {
        "left": int(win.left),
        "top": int(win.top),
        "width": int(win.width),
        "height": int(win.height),
    }

    raw = np.array(sct.grab(monitor))

    return cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2BGR,
    )


def crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    a = max(0, min(w, int(w * x1)))
    b = max(0, min(h, int(h * y1)))
    c = max(0, min(w, int(w * x2)))
    d = max(0, min(h, int(h * y2)))

    if c <= a or d <= b:
        return None

    return img[b:d, a:c]


def read_text(region):
    if region is None or region.size == 0:
        return ""

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    # One OCR pass only. Multiple OCR passes were freezing the Tkinter UI.
    try:
        return pytesseract.image_to_string(
            gray,
            config="--psm 6",
        ).strip()
    except Exception:
        return ""


def parse_distance_ft(text):
    upper = text.upper()

    ft = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*FT",
            upper,
        )
    ]

    if ft:
        return max(ft)

    yds = [
        float(x)
        for x in re.findall(
            r"(\d+(?:\.\d+)?)\s*YDS?",
            upper,
        )
    ]

    if yds:
        return max(yds) * 3.0

    return None


def parse_elevation_ft(text):
    upper = text.upper()

    values = [
        float(x)
        for x in re.findall(
            r"(-?\d+(?:\.\d+)?)\s*FT",
            upper,
        )
    ]

    if len(values) >= 2:
        return min(
            values,
            key=lambda v: abs(v),
        )

    return 0.0


def parse_green_speed(text):
    upper = text.upper()

    if "VERY FAST" in upper:
        return 11.5, "VERY FAST"

    if "FAST" in upper:
        return 10.0, "FAST"

    if "SLOW" in upper:
        return 7.0, "SLOW"

    return 9.0, "NORMAL"


def parse_player(text):
    candidates = re.findall(
        r"\b[A-Za-z][A-Za-z0-9_]{2,23}\b",
        text,
    )

    reject = {
        "lvl", "shot", "score", "hole", "tee",
        "green", "speed", "putter", "driver",
        "iron", "wedge", "yards", "feet",
    }

    for candidate in candidates:
        if candidate.lower() not in reject:
            return candidate

    return None


def detect_putting(text):
    upper = text.upper()
    return "PUTTER" in upper or "PUTT" in upper


def calculate_putt(distance_ft, elevation_ft, green_speed):
    if distance_ft is None:
        return None

    play = float(distance_ft)
    play += float(elevation_ft) * 1.8

    if green_speed >= 11.0:
        play *= 0.88
    elif green_speed >= 10.0:
        play *= 0.92
    elif green_speed <= 7.0:
        play *= 1.05

    return max(
        0.1,
        round(play, 1),
    )


def calculate_full_shot(distance_ft, elevation_ft):
    if distance_ft is None:
        return None

    yards = distance_ft / 3.0
    yards += float(elevation_ft) / 3.0

    return max(
        1.0,
        round(yards, 1),
    )


class WGTOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WGT Assistant")
        self.root.attributes("-topmost", True)
        self.root.geometry("420x340+50+50")
        self.root.configure(bg="#151515")
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.stop,
        )

        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.title = self.label(
            "WGT ASSISTANT",
            18,
            True,
        )

        self.player = self.label(
            "Player: waiting...",
            11,
        )

        self.mode = self.label(
            "Mode: --",
            13,
            True,
        )

        self.distance = self.label(
            "Distance: --",
            12,
        )

        self.elevation = self.label(
            "Elevation: --",
            12,
        )

        self.green = self.label(
            "Green speed: --",
            12,
        )

        self.calc = tk.Label(
            self.root,
            text="CALCULATION: waiting...",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#2a2a2a",
            padx=12,
            pady=14,
            wraplength=380,
        )
        self.calc.pack(
            fill="x",
            padx=14,
            pady=(12, 8),
        )

        self.status = tk.Label(
            self.root,
            text="Starting reader...",
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#151515",
            wraplength=380,
        )
        self.status.pack(
            padx=14,
            pady=(0, 10),
        )

        self.stop_button = tk.Button(
            self.root,
            text="Stop Assistant",
            command=self.stop,
            font=("Segoe UI", 10),
        )
        self.stop_button.pack(
            pady=(0, 12),
        )

        self.worker = threading.Thread(
            target=self.reader_worker,
            daemon=True,
        )
        self.worker.start()

        self.root.after(
            REFRESH_MS,
            self.process_results,
        )

    def label(self, text, size, bold=False):
        widget = tk.Label(
            self.root,
            text=text,
            font=(
                "Segoe UI",
                size,
                "bold" if bold else "normal",
            ),
            fg="white",
            bg="#151515",
        )
        widget.pack(pady=2)
        return widget

    def reader_worker(self):
        try:
            with mss.MSS() as sct:
                while not self.stop_event.is_set():
                    win = get_wgt_window()

                    if win is None:
                        self.result_queue.put({
                            "status": "Waiting for WGT Golf...",
                        })
                        time.sleep(1.0)
                        continue

                    try:
                        img = capture_wgt(
                            win,
                            sct,
                        )
                    except Exception as exc:
                        self.result_queue.put({
                            "status": f"Capture error: {exc}",
                        })
                        time.sleep(1.0)
                        continue

                    top_right = crop(
                        img,
                        0.72, 0.02,
                        1.00, 0.18,
                    )

                    center_top = crop(
                        img,
                        0.35, 0.06,
                        0.72, 0.36,
                    )

                    top_left = crop(
                        img,
                        0.00, 0.01,
                        0.36, 0.25,
                    )

                    bottom_left = crop(
                        img,
                        0.00, 0.76,
                        0.48, 1.00,
                    )

                    player_text = read_text(
                        top_right
                    )

                    center_text = read_text(
                        center_top
                    )

                    top_text = read_text(
                        top_left
                    )

                    bottom_text = read_text(
                        bottom_left
                    )

                    combined = "\n".join([
                        player_text,
                        center_text,
                        top_text,
                        bottom_text,
                    ])

                    player = parse_player(
                        player_text
                    )

                    putting = detect_putting(
                        combined
                    )

                    distance_ft = parse_distance_ft(
                        center_text
                    )

                    elevation_ft = parse_elevation_ft(
                        center_text
                    )

                    green_speed, green_name = parse_green_speed(
                        top_text + "\n" + bottom_text
                    )

                    if putting:
                        calculation = calculate_putt(
                            distance_ft,
                            elevation_ft,
                            green_speed,
                        )
                    else:
                        calculation = calculate_full_shot(
                            distance_ft,
                            elevation_ft,
                        )

                    self.result_queue.put({
                        "status": "Live",
                        "player": player,
                        "putting": putting,
                        "distance_ft": distance_ft,
                        "elevation_ft": elevation_ft,
                        "green_speed": green_speed,
                        "green_name": green_name,
                        "calculation": calculation,
                    })

                    time.sleep(
                        SCAN_DELAY
                    )

        except Exception as exc:
            self.result_queue.put({
                "status": f"Reader stopped: {exc}",
            })

    def process_results(self):
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.apply_result(
                    result
                )
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(
                REFRESH_MS,
                self.process_results,
            )

    def apply_result(self, result):
        status = result.get(
            "status",
            "Live",
        )

        self.status.config(
            text=status
        )

        if "putting" not in result:
            return

        player = result.get(
            "player"
        )

        putting = result["putting"]
        distance_ft = result.get(
            "distance_ft"
        )

        elevation_ft = result.get(
            "elevation_ft",
            0.0,
        )

        green_speed = result.get(
            "green_speed",
            9.0,
        )

        green_name = result.get(
            "green_name",
            "NORMAL",
        )

        calculation = result.get(
            "calculation"
        )

        self.player.config(
            text=f"Player: {player or 'reading...'}"
        )

        self.mode.config(
            text=(
                "Mode: PUTTING"
                if putting
                else "Mode: FULL SHOT"
            )
        )

        if distance_ft is None:
            self.distance.config(
                text="Distance: reading..."
            )
        elif putting:
            self.distance.config(
                text=f"Distance: {distance_ft:.1f} ft"
            )
        else:
            self.distance.config(
                text=f"Distance: {distance_ft / 3.0:.1f} yd"
            )

        self.elevation.config(
            text=f"Elevation: {elevation_ft:+.1f} ft"
        )

        self.green.config(
            text=(
                f"Green speed: {green_name} "
                f"({green_speed:.1f})"
            )
        )

        if calculation is None:
            self.calc.config(
                text="CALCULATION: READING HUD..."
            )
        elif putting:
            self.calc.config(
                text=f"PLAY PUTT: {calculation:.1f} ft"
            )
        else:
            self.calc.config(
                text=f"PLAY SHOT: {calculation:.1f} yd"
            )

    def stop(self):
        self.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if not os.path.exists(TESSERACT):
        raise FileNotFoundError(
            f"Tesseract not found: {TESSERACT}"
        )

    WGTOverlay().run()
