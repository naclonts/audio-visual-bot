import threading
import cv2, sys, time, os, pkg_resources
from picamera2 import Picamera2
from pantilthat import *
import anthropic
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder

load_dotenv()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/complete"
prompt_history = []
running = True  # Flag to control when threads should stop

# Face tracking variables
FRAME_W = 640
FRAME_H = 480
CAMERA_VERTICAL_FOV = 62.2
CAMERA_HORIZONTAL_FOV = 48.8
cam_pan = 0
cam_tilt = 90

def call_llm_api(prompt):
    global prompt_history
    client = anthropic.Anthropic()

    new_prompt_series = prompt_history + [
        {
            "role": "user",
            "content": prompt
        }
    ]

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are integrated into a robot that communicates through a Raspberry Pi device. Text from the robot's microphone is passed to you via the Anthropic API. You may also be passed some parsed visual cues as text. The robot has an integrated camera and face tracking device. Keep things short and conversational. Speak in the style of Thomas Carlyle. Note that because voice transcription is being done with a simple Whisper model before the text is passed to you, there may be some errors in the text transcription. Use your best guess as to the intention of the speaker.",
        messages=new_prompt_series
    )
    prompt_history = new_prompt_series + [
            {
                "role": "assistant",
                "content": message.content
            }
        ]

    return message.content

def handle_transcription(text):
    print(f"\nReal-time transcription: {text}\n")

    if len(prompt_history) == 0 and text == 'Thank you.': return
    if text.strip() == '': return

    response = call_llm_api(text)
    print(f"\nLLM Response: {response}")

def listen_to_audio():
    global running
    recorder = AudioToTextRecorder()
    recorder.start()
    try:
        while running:
            recorder.text(handle_transcription)
    except KeyboardInterrupt:
        pass  # Allow the main thread to handle the shutdown
    finally:
        recorder.stop()

def track_face():
    global cam_pan, cam_tilt, running

    haar_path = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(haar_path)
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": "XRGB8888", "size": (FRAME_W, FRAME_H)}))
    cam.start()
    time.sleep(1)

    pan(cam_pan)
    tilt(cam_tilt)

    while running:
        frame = cam.capture_array()
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)

        if len(faces) > 0:
            print(f'\n---- Found {len(faces)} faces ----')

        for (x, y, w, h) in faces:
            x = x + (w / 2)
            y = y + (h / 2)
            relative_x = float(x / FRAME_W) - 0.5
            relative_y = float(y / FRAME_H) - 0.5
            angle_horizontal = relative_x * CAMERA_HORIZONTAL_FOV
            angle_vertical = relative_y * CAMERA_VERTICAL_FOV

            cam_pan = get_pan() + angle_horizontal
            cam_tilt = get_tilt() + angle_vertical

            cam_pan = max(-90, min(90, cam_pan))
            cam_tilt = max(-90, min(90, cam_tilt))

            pan(int(cam_pan))
            tilt(int(cam_tilt))

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        audio_thread = threading.Thread(target=listen_to_audio)
        face_thread = threading.Thread(target=track_face)

        audio_thread.start()
        face_thread.start()

        audio_thread.join()
        face_thread.join()
    except KeyboardInterrupt:
        print("\nGracefully stopping...")
        running = False  # Set the flag to stop the threads
        audio_thread.join()
        face_thread.join()
        print("Stopped all threads.")
