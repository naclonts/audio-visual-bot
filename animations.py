from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
import math
import time

class AnimationHandler:
    def __init__(self, state):
        self.state = state
        self.serial = i2c(port=1, address=0x3C)
        self.device = ssd1306(self.serial, width=128, height=32)
        print(f"Display size: {self.device.width}x{self.device.height}")

    def draw_thinking(self):
        """Draw dots blinking in sequence for 'Thinking' state."""
        width = self.device.width
        height = self.device.height
        dot_positions = [
            (width // 2 - 20, height // 2),
            (width // 2, height // 2),
            (width // 2 + 20, height // 2)
        ]
        i = 0
        while self.state.value == "thinking":
            with canvas(self.device) as draw:
                for j in range(i % 4):
                    if j < len(dot_positions):
                        draw.ellipse(
                            (
                                dot_positions[j][0] - 2,
                                dot_positions[j][1] - 2,
                                dot_positions[j][0] + 2,
                                dot_positions[j][1] + 2
                            ),
                            fill=255
                        )
            i += 1
            time.sleep(0.5)  # Controls the blinking speed

    def draw_speaking(self, intensity=1):
        """Draw animated waveform for 'Speaking' state."""
        width = self.device.width
        height = self.device.height
        offset = 0  # This will create the moving effect

        while self.state.value == "speaking":
            with canvas(self.device) as draw:
                # Generate the waveform by shifting the sine wave horizontally (using `offset`)
                for x in range(width):
                    y = int((height // 2) + (math.sin((x + offset) / 10.0) * (10 * intensity)))
                    draw.line((x, height // 2, x, y), fill=255)

            # Increase the offset to make the waveform "move" over time
            offset += 5
            if offset > width:
                offset = 0

            # Adjust the delay to control the speed of the animation
            time.sleep(0.05)

    def draw_listening(self):
        """Draw a static circle for 'Listening' state."""
        width = self.device.width
        height = self.device.height
        radius = 15  # Fixed radius

        with canvas(self.device) as draw:
            # Draw a circle in the center of the screen
            draw.ellipse(
                (
                    width // 2 - radius,
                    height // 2 - radius,
                    width // 2 + radius,
                    height // 2 + radius
                ),
                outline=255,
                fill=0
            )

    def run(self):
        """Main loop to handle animations based on the current state."""
        while True:
            current_state = self.state.value
            if current_state == "thinking":
                self.draw_thinking()
            elif current_state == "speaking":
                self.draw_speaking()
            elif current_state == "listening":
                self.draw_listening()
                time.sleep(1)  # Static display, refresh every second
            else:
                # idle state
                print(f"state: {current_state}")
                time.sleep(1)

def start_animation_process(state):
    """Function to start the animation handler."""
    handler = AnimationHandler(state)
    handler.run()
