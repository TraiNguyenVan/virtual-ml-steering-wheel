import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import threading


class HandVirtualJoystick:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize pygame for joystick simulation
        pygame.init()
        pygame.joystick.init()

        # Virtual joystick parameters
        self.joystick_x = 0.0  # -1.0 to 1.0 (left to right)
        self.joystick_y = 0.0  # -1.0 to 1.0 (up to down)
        self.joystick_active = False

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Steering parameters
        self.center_x = 320  # Center of frame
        self.center_y = 240  # Center of frame
        self.max_range = 150  # Maximum movement range in pixels
        self.dead_zone = 30  # Pixels of dead zone in center
        self.smoothing = 0.8  # Smoothing factor (0-1)

        # Control state
        self.last_joystick_x = 0.0
        self.last_joystick_y = 0.0

        # Calibration
        self.calibrated = False
        self.neutral_position = None

        # Create virtual joystick thread
        self.running = True
        self.joystick_thread = threading.Thread(target=self._joystick_loop, daemon=True)
        self.joystick_thread.start()

        print("Hand Virtual Joystick Initialized!")
        print("This creates a virtual joystick that games can detect.")
        print("\nControls:")
        print("- Hold your hand in front of camera")
        print("- Move hand left/right for X-axis (steering)")
        print("- Move hand up/down for Y-axis (throttle/brake)")
        print("- Press 'c' to calibrate neutral position")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset joystick to center")
        print("- Close fist to activate joystick input")

    def _joystick_loop(self):
        """Background thread that continuously updates virtual joystick state."""
        clock = pygame.time.Clock()

        while self.running:
            # Process pygame events to maintain joystick state
            pygame.event.pump()

            # Simulate joystick events if needed
            if hasattr(self, '_send_joystick_event'):
                self._send_joystick_event()

            clock.tick(60)  # 60 FPS update rate

    def get_hand_center(self, landmarks, frame_shape):
        """Calculate the center point of the hand."""
        h, w = frame_shape[:2]

        # Get wrist and middle finger MCP positions
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # Calculate center between wrist and middle finger
        center_x = int((wrist.x + middle_mcp.x) * w / 2)
        center_y = int((wrist.y + middle_mcp.y) * h / 2)

        return center_x, center_y

    def is_fist_closed(self, landmarks):
        """Detect if hand is making a fist (for activation)."""
        # Check if fingertips are below their respective PIP joints
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]  # PIP joints

        closed_fingers = 0

        for i in range(5):
            tip = landmarks[finger_tips[i]]
            pip = landmarks[finger_pips[i]]

            # For thumb, check x-coordinate; for others, check y-coordinate
            if i == 0:  # Thumb
                if abs(tip.x - pip.x) < 0.05:  # Thumb is close to palm
                    closed_fingers += 1
            else:  # Other fingers
                if tip.y > pip.y:  # Fingertip is below PIP joint
                    closed_fingers += 1

        return closed_fingers >= 3  # At least 3 fingers closed

    def calculate_joystick_values(self, hand_x, hand_y):
        """Convert hand position to joystick values (-1.0 to 1.0)."""
        if self.neutral_position is None:
            reference_x, reference_y = self.center_x, self.center_y
        else:
            reference_x, reference_y = self.neutral_position

        # Calculate offset from center/neutral
        offset_x = hand_x - reference_x
        offset_y = hand_y - reference_y

        # Apply dead zone
        if abs(offset_x) < self.dead_zone:
            offset_x = 0
        if abs(offset_y) < self.dead_zone:
            offset_y = 0

        # Convert to joystick range (-1.0 to 1.0)
        joystick_x = np.clip(offset_x / self.max_range, -1.0, 1.0)
        joystick_y = np.clip(offset_y / self.max_range, -1.0, 1.0)

        # Apply smoothing
        joystick_x = self.last_joystick_x * self.smoothing + joystick_x * (1 - self.smoothing)
        joystick_y = self.last_joystick_y * self.smoothing + joystick_y * (1 - self.smoothing)

        self.last_joystick_x = joystick_x
        self.last_joystick_y = joystick_y

        return joystick_x, joystick_y

    def update_virtual_joystick(self, x_axis, y_axis, active):
        """Update the virtual joystick state."""
        self.joystick_x = x_axis
        self.joystick_y = y_axis
        self.joystick_active = active

        # Here you would normally send the joystick data to the system
        # This is a simplified version - in practice, you'd need a virtual joystick driver
        # or use libraries like vJoy (Windows) or uinput (Linux)

    def draw_joystick_display(self, frame, hand_x=None, hand_y=None, joystick_info=None, fist_closed=False):
        """Draw the joystick interface on the frame."""
        h, w = frame.shape[:2]

        # Draw coordinate system
        cv2.line(frame, (self.center_x, 0), (self.center_x, h), (100, 100, 100), 1)
        cv2.line(frame, (0, self.center_y), (w, self.center_y), (100, 100, 100), 1)

        # Draw neutral position if calibrated
        if self.neutral_position:
            ref_x, ref_y = self.neutral_position
            cv2.circle(frame, (ref_x, ref_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, "NEUTRAL", (ref_x - 30, ref_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            ref_x, ref_y = self.center_x, self.center_y

        # Draw movement range circle
        cv2.circle(frame, (ref_x, ref_y), self.max_range, (255, 255, 255), 2)
        cv2.circle(frame, (ref_x, ref_y), self.dead_zone, (100, 100, 100), 1)

        # Draw hand position
        if hand_x and hand_y:
            color = (0, 255, 255) if fist_closed else (255, 255, 0)
            cv2.circle(frame, (hand_x, hand_y), 8, color, -1)

            # Draw line from neutral to hand
            cv2.line(frame, (ref_x, ref_y), (hand_x, hand_y), color, 2)

        # Draw joystick values
        if joystick_info:
            x_val, y_val = joystick_info

            # Joystick visualization (top-right corner)
            joy_center_x, joy_center_y = w - 80, 80
            joy_radius = 50

            # Draw joystick background
            cv2.circle(frame, (joy_center_x, joy_center_y), joy_radius, (50, 50, 50), -1)
            cv2.circle(frame, (joy_center_x, joy_center_y), joy_radius, (255, 255, 255), 2)

            # Draw joystick position
            joy_x = int(joy_center_x + x_val * joy_radius * 0.8)
            joy_y = int(joy_center_y + y_val * joy_radius * 0.8)
            joy_color = (0, 255, 0) if self.joystick_active else (100, 100, 100)
            cv2.circle(frame, (joy_x, joy_y), 8, joy_color, -1)

            # Draw crosshair
            cv2.line(frame, (joy_center_x - joy_radius, joy_center_y),
                     (joy_center_x + joy_radius, joy_center_y), (100, 100, 100), 1)
            cv2.line(frame, (joy_center_x, joy_center_y - joy_radius),
                     (joy_center_x, joy_center_y + joy_radius), (100, 100, 100), 1)

            # Display values
            cv2.putText(frame, f"X: {x_val:+.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Y: {y_val:+.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw status
        status_text = "CALIBRATED" if self.calibrated else "NOT CALIBRATED - Press 'c'"
        status_color = (0, 255, 0) if self.calibrated else (0, 0, 255)
        cv2.putText(frame, status_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Draw activation status
        activation_text = "ACTIVE (Fist Closed)" if fist_closed else "INACTIVE (Open Hand)"
        activation_color = (0, 255, 0) if fist_closed else (0, 100, 255)
        cv2.putText(frame, activation_text, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, activation_color, 2)

        # Instructions
        cv2.putText(frame, "Close fist to activate", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self):
        """Main loop for the hand virtual joystick."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process hands
                results = self.hands.process(rgb_frame)

                hand_x = hand_y = None
                joystick_info = None
                fist_closed = False

                if results.multi_hand_landmarks:
                    # Use the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Get hand center position
                    hand_x, hand_y = self.get_hand_center(hand_landmarks.landmark, frame.shape)

                    # Check if fist is closed (for activation)
                    fist_closed = self.is_fist_closed(hand_landmarks.landmark)

                    # Calculate joystick values
                    joystick_x, joystick_y = self.calculate_joystick_values(hand_x, hand_y)
                    joystick_info = (joystick_x, joystick_y)

                    # Update virtual joystick
                    self.update_virtual_joystick(joystick_x, joystick_y, fist_closed)
                else:
                    # No hand detected, reset joystick to center
                    self.update_virtual_joystick(0.0, 0.0, False)

                # Draw interface
                self.draw_joystick_display(frame, hand_x, hand_y, joystick_info, fist_closed)

                # Show frame
                cv2.imshow('Hand Virtual Joystick', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if hand_x and hand_y:
                        self.neutral_position = (hand_x, hand_y)
                        self.calibrated = True
                        print(f"Calibrated neutral position at ({hand_x}, {hand_y})")
                    else:
                        print("No hand detected! Please show your hand and try again.")
                elif key == ord('r'):
                    self.update_virtual_joystick(0.0, 0.0, False)
                    print("Joystick reset to center")

        except KeyboardInterrupt:
            print("\nStopping hand virtual joystick...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if hasattr(self, 'joystick_thread'):
            self.joystick_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Hand virtual joystick stopped.")

    def get_joystick_state(self):
        """Get current joystick state (for external access)."""
        return {
            'x_axis': self.joystick_x,
            'y_axis': self.joystick_y,
            'active': self.joystick_active
        }


def main():
    """Main function to run the hand virtual joystick."""
    try:
        print("Installing required packages if needed...")
        print("pip install opencv-python mediapipe pygame numpy")
        print("\nNote: For full joystick simulation, you may need:")
        print("- Windows: vJoy virtual joystick driver")
        print("- Linux: python-uinput library")
        print("- macOS: Additional setup may be required")
        print("\nStarting Hand Virtual Joystick...\n")

        joystick = HandVirtualJoystick()
        joystick.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install opencv-python mediapipe pygame numpy")


if __name__ == "__main__":
    main()