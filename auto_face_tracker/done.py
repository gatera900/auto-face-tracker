#!/usr/bin/env python3
import cv2
import time
import numpy as np
import sys
import serial
# -------------------------- CONFIG --------------------------
CASCADE_PATH = './haarcascade_frontalface_default.xml'
CAM_IDX = 0
FRAME_SIZE = (800, 600)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SERIAL_PORT = 'COM13'
BAUD_RATE = 9600
# --- Motor tuning ---
MAX_ROTATE_STEP = 10       # max degrees per command
THRESHOLD = 20             # minimum pixels offset to start correction
SMOOTH_FACTOR = 0.04       # smaller = smoother (0.03–0.1 is good)
SEND_INTERVAL = 0.05        # seconds between commands
# -------------------------------------------------------------
def main():
    arduino = None
    cap = None
    try:
        # ---- Connect to Arduino ----
        try:
            arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"[OK] Connected to Arduino on {SERIAL_PORT}")
        except Exception as e:
            print(f"[WARN] Could not connect to Arduino: {e}")
            print("[INFO] Continuing without Arduino connection...")
            arduino = None
        
        # ---- Load cascade ----
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if cascade.empty():
            print(f"[ERROR] Could not load cascade: {CASCADE_PATH}")
            print("[INFO] Run download_cascade.py to download the cascade file")
            return
        
        # ---- Open camera ----
        cap = cv2.VideoCapture(CAM_IDX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return
        
        frame_center_x = FRAME_SIZE[0] // 2
        last_send = 0
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed.")
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                cx = x + w // 2
                offset = cx - frame_center_x
                # Draw face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, y + h // 2), 5, (0, 0, 255), -1)
                # ---- Proportional smooth rotation ----
                if abs(offset) > THRESHOLD and (time.time() - last_send > SEND_INTERVAL):
                    # Map offset to rotation angle (proportional control)
                    rotate_deg = np.clip(abs(offset) * SMOOTH_FACTOR, 1, MAX_ROTATE_STEP)
                    if offset < 0:
                        command = f"CCW {rotate_deg:.1f}\n"
                        print(f"Left ({offset}) → CCW {rotate_deg:.1f}°")
                    else:
                        command = f"CW {rotate_deg:.1f}\n"
                        print(f"Right ({offset}) → CW {rotate_deg:.1f}°")
                    if arduino:
                        try:
                            arduino.write(command.encode('utf-8'))
                        except Exception as e:
                            print(f"[WARN] Failed to send command to Arduino: {e}")
                    last_send = time.time()
            
            # Draw center line (optional)
            # cv2.line(frame, (frame_center_x, 0), (frame_center_x, FRAME_SIZE[1]), (255, 0, 0), 2)
            cv2.imshow("Smooth Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if arduino is not None:
            try:
                arduino.close()
                print("[INFO] Arduino connection closed")
            except:
                pass
        print("[INFO] Exiting...")

if __name__ == "__main__":
    main()

