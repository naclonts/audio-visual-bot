import anthropic
from dotenv import load_dotenv
from time import sleep
from RealtimeSTT import AudioToTextRecorder

load_dotenv()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/complete"

prompt_history = []

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

# Define callback functions
def handle_transcription(text):
    global recorder
    print(f"\nReal-time transcription: {text}\n")

    # there's some bugginess where "Thank you" gets printed at the beginning of every convo
    if len(prompt_history) == 0 and text == 'Thank you.': return

    if text.strip() == '': return

    #response = call_llm_api(text)
    #print(f"\nLLM Response: {response}")
    sleep(1)
    if text == 'Stop.':
        print("Stopping.")
        recorder.stop()
        return


# Initialize the recorder with real-time transcription enabled
recorder = AudioToTextRecorder()
recorder.start()

# Start recording and transcribin
try:
    while True:
       recorder.text(handle_transcription)
except KeyboardInterrupt:
    print("\nRecording stopped by user.")
