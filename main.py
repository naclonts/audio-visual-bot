import multiprocessing
import cv2, sys, time, os, pkg_resources
from picamera2 import Picamera2
from pantilthat import *
import anthropic
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
import wave
from piper.voice import PiperVoice
import queue
import pyaudio

load_dotenv()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/complete"
prompt_history = []
running = multiprocessing.Value('b', True)  # Use a multiprocessing.Value for running
audio_queue = multiprocessing.Queue()  # Queue to manage TTS audio playback

# Piper TTS Setup
voicedir = os.path.expanduser('~/Documents/piper/')
model = voicedir + "en_GB-northern_english_male-medium.onnx"
voice = PiperVoice.load(model)
is_playing_audio = multiprocessing.Value('b', False)  # Use a multiprocessing.Value for shared memory

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

def text_to_speech(text):
    """Convert text to speech and play it back."""
    sentences = text.split('. ')

    for i, sentence in enumerate(sentences):
        if sentence.strip():
            wav_file_path = f'output_{i}.wav'
            with wave.open(wav_file_path, 'w') as wav_file:
                voice.synthesize(sentence.strip(), wav_file)
            audio_queue.put(wav_file_path)

def audio_player(is_playing_audio, running):
    """Play audio files from the queue sequentially."""
    p = pyaudio.PyAudio()

    try:
        while running.value or not audio_queue.empty():
            if not audio_queue.empty():
                # Set the event to indicate audio is playing
                is_playing_audio.value = True

                wav_file_path = audio_queue.get()
                wf = wave.open(wav_file_path, 'rb')

                # Open a stream
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )

                # Read data in chunks
                data = wf.readframes(1024)

                # Play the sound by writing the audio data to the stream
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)

                # Stop and close the stream
                stream.stop_stream()
                stream.close()

                # Close the file
                wf.close()

                # Remove the wav file after playing
                os.remove(wav_file_path)

            # Clear the event when audio finishes playing
            if audio_queue.empty():
                is_playing_audio.value = False

    finally:
        p.terminate()  # Make sure PyAudio is properly terminated
        is_playing_audio.value = False  # Ensure the flag is clear if the loop ends

def handle_transcription(text):
    print(f"\nReal-time transcription: {text}\n")

    # don't get another response while the audio from the previous response is playing
    if is_playing_audio.value:
        return

    # there's some bugginess where "Thank you" gets transcribed during periods of silence
    if text =='Thank you.': return
    if text.strip() == '': return

    response = call_llm_api(text)
    print(f"\nLLM Response: {response}")
    text_to_speech('. '.join([r.text for r in response]))

def listen_to_audio(is_playing_audio, running):
    recorder = AudioToTextRecorder()
    recorder_started = False  # Track whether the recorder has started

    try:
        while running.value:
            if is_playing_audio.value:
                if recorder_started:
                    print("Stopping recorder.")
                    recorder.stop()  # Explicitly stop the recorder if audio is playing
                    recorder_started = False  # Update the flag since the recorder is stopped
            else:
                if not recorder_started:
                    print("Starting recorder.")
                    recorder.start()  # Start the recorder if it hasn't been started yet
                    recorder_started = True  # Update the flag since the recorder has started
                recorder.text(handle_transcription)

            time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in listen_to_audio")
    finally:
        if recorder_started:
            recorder.stop()  # Ensure the recorder is stopped on exit
        print("Audio recorder stopped.")

def track_face(running):
    global cam_pan, cam_tilt

    haar_path = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(haar_path)
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": "XRGB8888", "size": (FRAME_W, FRAME_H)}))
    cam.start()
    time.sleep(1)

    pan(cam_pan)
    tilt(cam_tilt)

    try:
        while running.value:
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
    finally:
        cam.stop()  # Properly stop the camera when the thread is ending

if __name__ == "__main__":
    try:
        p1 = multiprocessing.Process(target=listen_to_audio, args=(is_playing_audio, running))
        p2 = multiprocessing.Process(target=track_face, args=(running,))
        p3 = multiprocessing.Process(target=audio_player, args=(is_playing_audio, running))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()

    except KeyboardInterrupt:
        print("\nGracefully stopping...")
        running.value = False
        is_playing_audio.value = False  # Clear the flag if stopping
        print("Stopped all processes.")
