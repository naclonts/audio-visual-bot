import time
import multiprocessing
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import board
import busio
from adafruit_pca9685 import PCA9685

# LED Configuration
LED_CHANNEL_GREEN = 2  # Green LED
LED_CHANNEL_RED = 3  # Red LED

def set_led_brightness(channel, brightness, pca):
    """
    Set the brightness of an LED.

    Parameters:
        channel (int): The PCA9685 channel number connected to the LED.
        brightness (float): Brightness level from 0.0 (off) to 1.0 (full brightness).
        pca (PCA9685): The PCA9685 instance.
    """
    # Ensure brightness is within bounds
    brightness = max(0.0, min(brightness, 1.0))

    # PCA9685 has 16-bit resolution (0-65535)
    pwm_value = int(brightness * 65535)
    pca.channels[channel].duty_cycle = pwm_value

def initialize_leds():
    """Initialize the I2C bus and PCA9685 for LED control."""
    # Initialize I2C bus.
    i2c = busio.I2C(board.SCL, board.SDA)

    # Initialize PCA9685 using the I2C bus
    pca = PCA9685(i2c)
    pca.frequency = 1000  # Higher frequency for LEDs

    return pca

def perform_sentiment_analysis(text, tokenizer, model):
    """
    Perform sentiment analysis on the given text.

    Parameters:
        text (str): The text to analyze.
        tokenizer: The tokenizer instance.
        model: The sentiment analysis model.

    Returns:
        str: 'positive', 'negative', or 'neutral'
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    sentiment = model.config.id2label[predicted_class_id].lower()
    return sentiment

def sentiment_led_handler(sentiment_queue, running):
    """
    Handle sentiment analysis and LED control.

    Parameters:
        sentiment_queue (multiprocessing.Queue): Queue to receive LLM responses.
        running (multiprocessing.Value): Shared value to control the running state.
    """
    # Initialize sentiment analysis model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Initialize LEDs
    pca = initialize_leds()

    try:
        while running.value:
            try:
                # Wait for a new message with a timeout to allow checking the running flag
                message = sentiment_queue.get(timeout=0.5)
            except:
                continue  # Timeout occurred, loop back to check running flag

            sentiment = perform_sentiment_analysis(message, tokenizer, model)
            print(f"Sentiment Analysis Result: {sentiment}")

            if sentiment == 'positive':
                # Light Green LED
                print("Lighting Green LED")
                set_led_brightness(LED_CHANNEL_GREEN, 1.0, pca)  # Full brightness
                set_led_brightness(LED_CHANNEL_RED, 0.0, pca)  # Off
            elif sentiment == 'negative':
                # Light Red LED
                print("Lighting Red LED")
                set_led_brightness(LED_CHANNEL_GREEN, 0.0, pca)  # Off
                set_led_brightness(LED_CHANNEL_RED, 1.0, pca)  # Full brightness
            else:
                # Optional: Handle neutral or other sentiments
                print("Lighting Red and Green LEDs at 25% brightness")
                set_led_brightness(LED_CHANNEL_GREEN, 0.25, pca)
                set_led_brightness(LED_CHANNEL_RED, 0.25, pca)

            # Optional: Add a delay or keep the LED on for a certain duration
            time.sleep(2)

    except KeyboardInterrupt:
        print("Sentiment LED handler interrupted by user.")
    finally:
        # Turn off LEDs before exiting
        set_led_brightness(LED_CHANNEL_GREEN, 0.0, pca)
        set_led_brightness(LED_CHANNEL_RED, 0.0, pca)
        pca.deinit()
        print("Sentiment LED handler terminated gracefully.")

def start_sentiment_led_process(sentiment_queue, running):
    """Start the sentiment LED handler process."""
    sentiment_process = multiprocessing.Process(
        target=sentiment_led_handler,
        args=(sentiment_queue, running),
        name="SentimentLEDHandler"
    )
    sentiment_process.start()
    return sentiment_process
