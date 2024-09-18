This repo contains the code for a conversational robot toy. 

The capabilities of the robot, as of this commit, are:

1. Speech to text conversion of microphone audio in English to text. (Whisper)
2. Sending the text to an LLM and getting a conversational response. (Claude)
3. Converting the LLM's response to audio and playing it. (ElevenLabs)
4. Performing sentiment analysis on the LLM response and lighting a green or red LED for positive or negative sentiment. (DistilBERT)
5. Animating a small OLED display to illustrate whether the robot is currently listening, thinking, or speaking.
6. Based on camera input, locating any faces in the frame and moving pan/tilt servos to point at the face. (OpenCV, Haar cascade)
