import cv2
import torch
import asyncio
import time
import tkinter as tk
import httpx
from ultralytics import YOLO
from telegram import Bot
from telegram.request import HTTPXRequest

# ==========================================
# 1. Configuration & Parameters
# ==========================================
# Replace with your own Telegram Bot Token and Chat ID
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

MODEL_PATH = "yolo11n.pt"  # Using default YOLOv11 nano model
CONFIDENCE_THRESHOLD = 0.5
NOTIFICATION_DELAY = 30    # Cooldown period in seconds between alerts

last_notify_time = 0

# Configure Telegram Bot Request Timeout to prevent hanging
request_config = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
bot = Bot(token=TELEGRAM_TOKEN, request=request_config)

# Initialize Hidden Tkinter Root Window for Non-blocking Popups
root = tk.Tk()
root.withdraw()


# ==========================================
# 2. GUI Alert Popup Function
# ==========================================
def show_alert_popup(label):
    """
    Creates a non-blocking GUI alert window on top of other applications.
    """
    try:
        popup = tk.Toplevel(root)
        popup.title("⚠️ SECURITY ALERT")
        popup.geometry("400x150")
        popup.configure(bg="#ffcccc")
        popup.attributes('-topmost', True)

        msg = tk.Label(
            popup, 
            text=f"🚨 DETECTED: {label.upper()} 🚨",
            font=("Arial", 18, "bold"), 
            bg="#ffcccc", 
            fg="red"
        )
        msg.pack(expand=True)

        status_lbl = tk.Label(
            popup, 
            text="Alert notification sent to Telegram.",
            font=("Arial", 10), 
            bg="#ffcccc"
        )
        status_lbl.pack(pady=5)

        # Auto-destroy window after 5 seconds (5000 ms)
        popup.after(5000, popup.destroy)
        print(f"🖥️ [GUI Alert]: Displayed for '{label}'")
    except Exception as e:
        print(f"❌ [GUI Error]: {e}")


# ==========================================
# 3. Async Telegram Alert Function
# ==========================================
async def send_telegram_alert(label):
    """
    Sends an asynchronous message to the configured Telegram chat.
    """
    try:
        print(f"🚀 [Async] Sending Telegram alert for: {label}...")
        await bot.send_message(chat_id=CHAT_ID, text=f"🔔 Detection Alert: {label}")
        print("✅ [Async] Telegram message sent successfully!")
    except Exception as err:
        print(f"❌ [Telegram Error]: {err}")


# ==========================================
# 4. Main Application Loop
# ==========================================
async def main():
    global last_notify_time

    print("--- Starting Smart Object Detection System ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️ Running on Device: {device.upper()}")

    # Load Object Detection Model
    try:
        print(f"📦 Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH).to(device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Check Telegram Connection Status
    print("📧 Testing Telegram connection...")
    try:
        await bot.send_message(chat_id=CHAT_ID, text="✅ Smart Detection System Started")
        print("✅ Telegram bot initialized successfully.")
    except Exception as e:
        print(f"⚠️ Telegram Initialization Warning: {e}")
        print("⚠️ System will continue to run without live notifications.")

    # Initialize Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    print("🎥 Video stream active. Press 'q' to quit.")

    while True:
        # Update Tkinter GUI event loop
        root.update()
        
        # Yield control to allow asynchronous tasks to execute
        await asyncio.sleep(0.01)

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera.")
            break

        # Run Object Detection Inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, stream=True, device=device)

        for r in results:
            annotated_frame = r.plot()

            if len(r.boxes) > 0:
                current_time = time.time()

                # Trigger Notification Cooldown Check
                if current_time - last_notify_time > NOTIFICATION_DELAY:
                    class_id = int(r.boxes[0].cls)
                    label = model.names[class_id]
                    confidence = float(r.boxes[0].conf[0])

                    print(f"⚠️ Detected: {label} ({confidence:.2f}) -> Triggering Alerts")

                    # Dispatch Asynchronous Telegram Alert Task
                    asyncio.create_task(send_telegram_alert(label))

                    # Trigger Local GUI Popup Window
                    show_alert_popup(label)

                    last_notify_time = current_time

            # Render Monitor Stream
            cv2.imshow("Smart Detection Monitor", annotated_frame)

        # Press 'q' on the OpenCV Window to Terminate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup Resources
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    print("🛑 System shutdown complete.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user.")