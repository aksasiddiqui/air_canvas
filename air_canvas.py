import cv2
import mediapipe as mp
import numpy as np
import time
import os

class AirCanvas:
    def __init__(self):
        # ---------------- MediaPipe Setup ----------------
        # Initialize MediaPipe Hands module to detect and track hand landmarks
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          # Real-time mode
            max_num_hands=1,                  # Only track 1 hand
            min_detection_confidence=0.7,     # Confidence threshold for detection
            min_tracking_confidence=0.7       # Confidence threshold for tracking
        )
        self.mp_draw = mp.solutions.drawing_utils  # Utility to draw hand landmarks

        # ---------------- Drawing State ----------------
        self.drawing_enabled = True  # True if drawing mode is ON
        self.prev_point = None       # Previous finger point to draw smooth line
        self.brush_thickness = 5     # Thickness of brush

        # ---------------- Colors ----------------
        # List of 7 basic colors (darker for palette)
        # Each color is a tuple (Name, BGR)
        self.colors = [
            ("RED",     (0, 0, 150)),
            ("GREEN",   (0, 150, 0)),
            ("BLUE",    (150, 0, 0)),
            ("YELLOW",  (0, 150, 150)),
            ("CYAN",    (150, 150, 0)),
            ("MAGENTA", (150, 0, 150)),
            ("BLACK",   (0, 0, 0))
        ]
        self.drawing_color = self.colors[1][1]  # Default color: GREEN

        # ---------------- Palette State ----------------
        self.palette_open = False           # True when palette is visible
        self.prev_fist_state = False       # To detect fist toggle

        # ---------------- Open-Hand Toggle State ----------------
        self.prev_open_hand_state = False  # To detect open-hand toggle

        # ---------------- Canvas ----------------
        self.canvas = None                  # Blank canvas for drawing

        # ---------------- Save Folder ----------------
        os.makedirs("saved_drawings", exist_ok=True)  # Folder to save PNGs

    # ---------------- Hand Finger Checks ----------------
    def fingers_up(self, lm):
        """
        Check which fingers are up.
        Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
        True if finger is up.
        """
        fingers = []
        # Thumb: compare x-axis positions (left vs right)
        fingers.append(lm[4].x < lm[3].x)
        # Other fingers: tip above PIP joint
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for t, p in zip(tips, pips):
            fingers.append(lm[t].y < lm[p].y)
        return fingers

    # ---------------- Fist Detection ----------------
    def is_fist(self, lm):
        """Returns True if all fingers are down (fist)"""
        return not any(self.fingers_up(lm))

    # ---------------- Full Hand Open Detection ----------------
    def is_hand_open(self, lm):
        """Returns True if all fingers are up (hi-five)"""
        return all(self.fingers_up(lm))

    # ---------------- Draw Color Palette ----------------
    def draw_palette(self, frame):
        """
        Draw the color palette at top of the screen.
        Each box shows a color and label under it.
        """
        h, w, _ = frame.shape
        box_w = w // len(self.colors)   # Width of each color box
        box_h = 70                      # Height of color box

        # Draw each color box
        for i, (name, color) in enumerate(self.colors):
            x1 = i * box_w
            x2 = (i + 1) * box_w
            # Draw the rectangle for the color
            cv2.rectangle(frame, (x1, 0), (x2, box_h), color, -1)
            # Draw white border
            cv2.rectangle(frame, (x1, 0), (x2, box_h), (255, 255, 255), 2)
            # Draw color label under the box
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (box_w - text_size[0]) // 2
            cv2.putText(frame, name, (text_x, box_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ---------------- Color Selection ----------------
    def select_color(self, x, frame_width):
        """Select color based on x-coordinate of index finger"""
        box_w = frame_width // len(self.colors)
        index = x // box_w
        if index < len(self.colors):
            self.drawing_color = self.colors[index][1]

    # ---------------- Main Loop ----------------
    def run(self):
        # Open camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Initialize blank canvas same size as frame
        self.canvas = np.zeros_like(frame)

        # ---------------- Fullscreen ----------------
        cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Air Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            drawing_active = False

            if results.multi_hand_landmarks:
                # Get landmarks of first hand
                lm = results.multi_hand_landmarks[0].landmark
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

                # ---------------- FIST DETECTION → PALETTE ----------------
                fist_now = self.is_fist(lm)
                if fist_now and not self.prev_fist_state:
                    self.palette_open = not self.palette_open  # Toggle palette
                    # Pause drawing while palette is open
                    self.drawing_enabled = not self.palette_open
                    self.prev_point = None  # reset previous point
                self.prev_fist_state = fist_now

                # ---------------- OPEN HAND TOGGLE DRAWING ----------------
                open_now = self.is_hand_open(lm)
                if open_now and not self.prev_open_hand_state:
                    self.drawing_enabled = not self.drawing_enabled
                    self.prev_point = None
                self.prev_open_hand_state = open_now

                # ---------------- INDEX FINGER POSITION ----------------
                h, w, _ = frame.shape
                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)
                current = (ix, iy)

                # ---------------- PALETTE MODE ----------------
                if self.palette_open:
                    self.draw_palette(frame)
                    if self.fingers_up(lm)[1]:  # index finger up → select color
                        self.select_color(ix, w)

                # ---------------- DRAWING MODE ----------------
                else:
                    if self.drawing_enabled and self.fingers_up(lm)[1]:
                        drawing_active = True
                        if self.prev_point:
                            cv2.line(self.canvas, self.prev_point, current,
                                     self.drawing_color, self.brush_thickness)
                        self.prev_point = current
                    else:
                        self.prev_point = None

            else:
                # Reset if no hand detected
                self.prev_point = None
                self.prev_fist_state = False
                self.prev_open_hand_state = False

            # ---------------- Merge Canvas & Frame ----------------
            # Set full opacity for canvas on screen
            output = cv2.addWeighted(frame, 1, self.canvas, 1, 0)

            # ---------------- STATUS PANEL ----------------
            # Small transparent instruction bar at bottom (~3cm)
            h, w, _ = output.shape
            overlay = output.copy()
            instructions = "Fist - Palette | Open Hand - Toggle Drawing | C - Clear | S - Save | Q - Quit"

            # Make panel slightly taller to fit text
            panel_height = int(h * 0.06)  # slightly taller

            # Draw semi-transparent overlay
            cv2.rectangle(overlay, (0, h - panel_height), (w, h), (50, 50, 50), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            # Compute text size for vertical centering
            (text_w, text_h), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            text_y = h - panel_height + (panel_height + text_h) // 2

            # Draw text
            cv2.putText(output, instructions, (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            # ---------------- Current Color Indicator ----------------
            cv2.rectangle(output, (w - 120, 10), (w - 40, 80),
                          self.drawing_color, -1)
            cv2.rectangle(output, (w - 120, 10), (w - 40, 80), (255, 255, 255), 2)

            # ---------------- Show Output ----------------
            cv2.imshow("Air Canvas", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)  # Clear canvas
            elif key == ord('s'):
                filename = f"saved_drawings/drawing_{int(time.time())}.png"
                cv2.imwrite(filename, self.canvas)
                print("Saved:", filename)

        # ---------------- Cleanup ----------------
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    AirCanvas().run()
