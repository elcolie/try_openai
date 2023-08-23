import sounddevice as sd
import numpy as np
import time


def play_beep(frequency, duration, volume=0.5):
    # Generate a beep sound
    t = np.linspace(0, duration, int(duration * 44100), False)
    beep = np.sin(2 * np.pi * frequency * t)

    # Play the beep sound
    sd.play(volume * beep, 44100)
    sd.wait()


def main() -> None:
    # Play a beep sound at 440 Hz for 0.5 seconds
    play_beep(440, 0.5)

    # Wait for a moment
    time.sleep(1)

    # Play another beep sound at 880 Hz for 0.3 seconds
    play_beep(880, 0.3)


if __name__ == "__main__":
    main()
