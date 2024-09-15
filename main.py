import multiprocessing
import time, os, signal
from pantilthat import *
import anthropic
from dotenv import load_dotenv
from RealtimeSTT import AudioToTextRecorder
import wave
from piper.voice import PiperVoice
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

from object_tracking import get_object_tracking_processes

load_dotenv()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/complete"
prompt_history = []
running = multiprocessing.Value('b', True)  # Use a multiprocessing.Value for running
audio_queue = multiprocessing.Queue()  # Queue to manage TTS audio playback

# Piper TTS Setup
USE_LOCAL_TTS = False
if USE_LOCAL_TTS:
    voicedir = os.path.expanduser('~/Documents/piper/')
    model = voicedir + "en_GB-northern_english_male-medium.onnx"
    voice = PiperVoice.load(model)

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

    system_prompt = "The assistant is integrated into a robot that communicates through a Raspberry Pi device. " + \
        "Text from the robot's microphone is passed to the assistant via the Anthropic API. " + \
        "The assistant may also be passed some parsed visual cues as text. The robot has an integrated camera and face tracking device. " + \
        "The assistant thinks and speaks in the style of Thomas Carlyle. " + \
        "Keeps things short and conversational. Brevity is favored, to allow an interactive exchange. The assistant replies in one or two sentences unless a longer monologue is warranted. " + \
        "Note that because voice transcription is being done with a simple Whisper model before the text is passed to the assistant, there may be some errors in the text transcription. Buest guesses should be used as to the intention of the speaker."

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
        system=system_prompt,
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

    if USE_LOCAL_TTS:
        # Use the Piper TTS model
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                wav_file_path = f'output_{i}.wav'
                with wave.open(wav_file_path, 'w') as wav_file:
                    voice.synthesize(sentence.strip(), wav_file, sentence_silence=0.75)
                audio_queue.put(wav_file_path)

    else:
        # Use the Eleven Labs API
        voice_id = 'ZQe5CZNOzWyzPSCn5a3c' # "George"
        headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": os.getenv("ELEVEN_LABS_API_KEY"),
            "Content-Type": "application/json"
        }

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                file_path = f'output_{i}.mp3'

                payload = {
                    "text": sentence.strip(),
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.7
                    },
                    "model_id": "eleven_turbo_v2"
                }

                response = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    # Save the response content (audio data) to an MP3 file
                    with open(file_path, 'wb') as mp3_file:
                        mp3_file.write(response.content)

                    # Put the MP3 file path in the queue if needed
                    audio_queue.put(file_path)
                else:
                    print(f"Error: {response.status_code} - {response.text}")

def audio_player(context, running):
    """Play audio files from the queue sequentially."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    p = pyaudio.PyAudio()

    try:
        while running.value or not audio_queue.empty():
            if not audio_queue.empty():
                # Set the event to indicate audio is playing
                context.is_playing_audio = True

                audio_file_path = audio_queue.get()
                print(f"Playing audio file: {audio_file_path}. context.is_playing_audio: {context.is_playing_audio}")

                # Check the file extension
                file_extension = os.path.splitext(audio_file_path)[1].lower()

                if file_extension == '.wav':
                    # Handling WAV file playback
                    wf = wave.open(audio_file_path, 'rb')

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

                elif file_extension == '.mp3':
                    # Handling MP3 file playback using pydub
                    audio = AudioSegment.from_file(audio_file_path, format="mp3")
                    play(audio)

                # Remove the audio file after playing
                print(f"Removing audio file: {audio_file_path}. Queue empty: {audio_queue.empty()}")
                os.remove(audio_file_path)

            # Clear the event when audio finishes playing
            if audio_queue.empty():
                context.is_playing_audio = False

    finally:
        p.terminate()  # Make sure PyAudio is properly terminated
        context.is_playing_audio = False  # Ensure the flag is clear if the loop ends

def handle_transcription(context, text):
    print(f"\nReal-time transcription: {text}.\nis_playing_audio: {context.is_playing_audio}\n")

    # don't get another response while the audio from the previous response is playing
    if context.is_playing_audio:
        print("Audio is playing. Skipping transcription.")
        return

    # there's some bugginess where "Thank you" gets transcribed during periods of silence
    if text =='Thank you.': return
    if text.strip() == '': return

    response = call_llm_api(text)
    print(f"\nLLM Response: {response}")
    text_to_speech('. '.join([r.text for r in response]))

def listen_to_audio(context, running):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    recorder = AudioToTextRecorder()
    recorder_started = False  # Track whether the recorder has started

    def transcribe(text):
        return handle_transcription(context, text)

    try:
        while running.value:
            if context.is_playing_audio:
                if recorder_started:
                    print("Stopping recorder.")
                    recorder.stop()  # Explicitly stop the recorder if audio is playing
                    recorder_started = False  # Update the flag since the recorder is stopped
            else:
                if not recorder_started:
                    print("Starting recorder.")
                    recorder.start()  # Start the recorder if it hasn't been started yet
                    recorder_started = True  # Update the flag since the recorder has started
                recorder.text(transcribe)

            time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in listen_to_audio")
        if recorder_started:
            recorder.stop()  # Ensure the recorder is stopped on exit
        print("Audio recorder stopped.")
        raise KeyboardInterrupt

if __name__ == "__main__":
    try:
        manager = multiprocessing.Manager()
        context = manager.Namespace()
        context.is_playing_audio = False

        processes = [
            multiprocessing.Process(target=listen_to_audio, args=(context, running)),
            multiprocessing.Process(target=audio_player, args=(context, running)),
        ]

        processes += get_object_tracking_processes(manager)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    except KeyboardInterrupt:
        print("\nGracefully stopping...")
        running.value = False
        context.is_playing_audio = False  # Clear the flag if stopping
        print("Stopped all processes. Exiting.")
        time.sleep(0.5)
        os._exit(1)