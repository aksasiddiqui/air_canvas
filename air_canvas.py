import cv2
import mediapipe as mp
import numpy as np
import time
import os

class AirCanvas:
    def __init__(self):
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          
            max_num_hands=1,                 
            min_detection_confidence=0.7,   
            min_tracking_confidence=0.7       
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.drawing_enabled = True 
        self.prev_point = None       
        self.brush_thickness = 5   

        self.colors = [
            ("RED",     (0, 0, 150)),
            ("GREEN",   (0, 150, 0)),
            ("BLUE",    (150, 0, 0)),
            ("YELLOW",  (0, 150, 150)),
            ("CYAN",    (150, 150, 0)),
            ("MAGENTA", (150, 0, 150)),
            ("BLACK",   (0, 0, 0))
        ]
        self.drawing_color = self.colors[1][1]  

        self.palette_open = False    
        self.prev_fist_state = False     

        self.prev_open_hand_state = False  

        self.canvas = None              

        os.makedirs("saved_drawings", exist_ok=True) 

    def fingers_up(self, lm):
        """
        Check which fingers are up.
        Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
        True if finger is up.
        """
        fingers = []
        fingers.append(lm[4].x < lm[3].x)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for t, p in zip(tips, pips):
            fingers.append(lm[t].y < lm[p].y)
        return fingers

    def is_fist(self, lm):
        """Returns True if all fingers are down (fist)"""
        return not any(self.fingers_up(lm))

    def is_hand_open(self, lm):
        """Returns True if all fingers are up (hi-five)"""
        return all(self.fingers_up(lm))

    def draw_palette(self, frame):
        """
        Draw the color palette at top of the screen.
        Each box shows a color and label under it.
        """
        h, w, _ = frame.shape
        box_w = w // len(self.colors) 
        box_h = 70         

        for i, (name, color) in enumerate(self.colors):
            x1 = i * box_w
            x2 = (i + 1) * box_w
            cv2.rectangle(frame, (x1, 0), (x2, box_h), color, -1)
            cv2.rectangle(frame, (x1, 0), (x2, box_h), (255, 255, 255), 2)
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (box_w - text_size[0]) // 2
            cv2.putText(frame, name, (text_x, box_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def select_color(self, x, frame_width):
        """Select color based on x-coordinate of index finger"""
        box_w = frame_width // len(self.colors)
        index = x // box_w
        if index < len(self.colors):
            self.drawing_color = self.colors[index][1]

    def run(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        self.canvas = np.zeros_like(frame)

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
                lm = results.multi_hand_landmarks[0].landmark
                self.mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    self.mp_hands.HAND_CONNECTIONS
                )

                fist_now = self.is_fist(lm)
                if fist_now and not self.prev_fist_state:
                    self.palette_open = not self.palette_open 
                    self.drawing_enabled = not self.palette_open
                    self.prev_point = None 
                self.prev_fist_state = fist_now

                open_now = self.is_hand_open(lm)
                if open_now and not self.prev_open_hand_state:
                    self.drawing_enabled = not self.drawing_enabled
                    self.prev_point = None
                self.prev_open_hand_state = open_now

                h, w, _ = frame.shape
                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)
                current = (ix, iy)

                if self.palette_open:
                    self.draw_palette(frame)
                    if self.fingers_up(lm)[1]:
                        self.select_color(ix, w)

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
                self.prev_point = None
                self.prev_fist_state = False
                self.prev_open_hand_state = False

            output = cv2.addWeighted(frame, 1, self.canvas, 1, 0)

            h, w, _ = output.shape
            overlay = output.copy()
            instructions = "Fist - Palette | Open Hand - Toggle Drawing | C - Clear | S - Save | Q - Quit"

            panel_height = int(h * 0.06)

            cv2.rectangle(overlay, (0, h - panel_height), (w, h), (50, 50, 50), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            (text_w, text_h), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            text_y = h - panel_height + (panel_height + text_h) // 2

            cv2.putText(output, instructions, (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.rectangle(output, (w - 120, 10), (w - 40, 80),
                          self.drawing_color, -1)
            cv2.rectangle(output, (w - 120, 10), (w - 40, 80), (255, 255, 255), 2)

            cv2.imshow("Air Canvas", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
            elif key == ord('s'):
                filename = f"saved_drawings/drawing_{int(time.time())}.png"
                cv2.imwrite(filename, self.canvas)
                print("Saved:", filename)

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    AirCanvas().run()
