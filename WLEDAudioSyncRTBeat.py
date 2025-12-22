import pyaudio
import numpy as np
import aubio  # Ledfx fork for better accuracy
import time
import sys
import math
import ipaddress
import argparse
import signal
import os
from typing import List, NamedTuple, Tuple
from collections import deque

from pythonosc.udp_client import SimpleUDPClient
if sys.platform != 'darwin':
    import keyboard

CHANNELS = 1  # Mono audio
FORMAT = pyaudio.paFloat32  # 32-bit float format, ideal for aubio


class BeatPrinter:
    """A simple class to manage the state of a spinning character for printing."""

    def __init__(self):
        self.state: int = 0
        self.spinner_chars = "¼▚▞▚"

    def get_char(self) -> str:
        char = self.spinner_chars[self.state]
        self.state = (self.state + 1) % len(self.spinner_chars)
        return char


class ServerInfo(NamedTuple):
    ip: str
    port: int
    address: str
    mode: str = None


def list_devices(p: pyaudio.PyAudio):
    """Lists all available audio input devices."""
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            print(f"  [{info['index']}] {info['name']}")
    print("\nUse the index with the -d flag to select a device.")


class BeatDetector:
    def __init__(self, device_index: int = None, silence_threshold: float = -60.0, server_info: List[ServerInfo] = None,
                 confidence_threshold: float = 0.2, doubling_confidence_threshold: float = 0.5, buf_size: int = 1024,
                 raw_bpm_mode: bool = False, relearn_interval: int = 0, doubling_threshold: float = 100.0):
        self.device_index = device_index
        self.silence_threshold = silence_threshold
        self.server_info = server_info
        self.buf_size = buf_size
        self.samplerate = 44100  # Default, will be updated
        self.doubling_confidence_threshold = doubling_confidence_threshold
        self.raw_bpm_mode = raw_bpm_mode
        self.relearn_interval = relearn_interval
        self.doubling_threshold = doubling_threshold

        # --- State Management ---
        self.is_playing = False
        self.is_learning = False  # Flag for the initial BPM learning phase
        self.avg_db_level = -120.0
        self.last_bpm = 0.0
        self.last_raw_bpm = 0.0  # Store the last raw BPM for consistent printing
        self.bpm_history = deque(maxlen=5)  # History of recent BPMs to make smarter decisions
        self.bpm_history_raw = deque(maxlen=5)  # History of recent raw BPMs to make smarter decisions
        self.sound_counter = 0
        self.silence_counter = 0
        self.last_update_time = 0.0
        self.listening_start_time = 0.0
        self.learning_phase_beats = []  # Store initial beats to make a better first guess
        self.last_relearn_time = 0.0  # Track time for periodic re-learning
        # --- Sanity Check State ---
        self.sanity_check_beats = 0
        self.sanity_check_start_time = 0.0

        self.last_callback_time = time.time()  # For watchdog timer

        # --- Constants ---
        self.confidence_threshold = confidence_threshold
        self.bpm_smoothing_factor = 0.1
        self.sound_frames_needed = 3
        self.silence_frames_needed = 10
        self.update_interval = 0.5
        self.listening_timeout = 8.0  # Seconds to wait in "Listening" before resetting
        self.learning_beats_needed = 5  # Number of beats to collect before making a decision
        self.sanity_check_beats_needed = 10  # Number of beats to count for the sanity check
        self.sanity_check_window = 10.0  # Max seconds for the sanity check window
        self.watchdog_timeout = 2.0  # Seconds of no callbacks before forcing silence

        # --- Printing ---
        self.spinner = BeatPrinter()

        # --- OSC Client Setup ---
        self.osc_servers: List[Tuple[SimpleUDPClient, str]] = []
        if self.server_info:
            self.osc_servers = [(SimpleUDPClient(x.ip, x.port), x.address) for x in self.server_info]

        # --- PyAudio and Aubio Setup ---
        self.p = pyaudio.PyAudio()

        # Query the device for its default sample rate
        device_info = self.p.get_device_info_by_index(self.device_index)
        self.samplerate = int(device_info['defaultSampleRate'])
        print(f"Using device sample rate: {self.samplerate} Hz")

        fft_size = self.buf_size * 2
        # Use the 'specflux' method from the ledfx fork for potentially better accuracy
        self.tempo = aubio.tempo("specflux", fft_size, self.buf_size, self.samplerate)
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.samplerate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.buf_size,
            stream_callback=self._pyaudio_callback
        )

    def trigger_relearn(self):
        """Manually triggers the BPM learning phase."""
        # Only trigger if music is playing and we are not already learning
        if self.is_playing and not self.is_learning:
            self.is_learning = True
            self.listening_start_time = time.time()
            self.learning_phase_beats.clear()
            print(f"\n[Manual Trigger] Re-learning BPM... (current: {self.last_bpm:.1f})")

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        try:
            is_watchdog_timeout = (time.time() - self.last_callback_time) > self.watchdog_timeout
            # Update watchdog timer
            self.last_callback_time = time.time()

            audio_samples = np.frombuffer(in_data, dtype=np.float32)
            db_level = aubio.db_spl(audio_samples)

            if db_level == -np.inf:
                db_level = -120.0

            self.avg_db_level = (0.2 * self.avg_db_level) + (0.8 * db_level)

            # --- CRITICAL: Sanitize the average level to recover from NaN ---
            # This can happen with unusual audio signals.
            if np.isnan(self.avg_db_level):
                self.avg_db_level = -120.0

            # --- State Machine Logic ---
            # Decide if we are playing or silent based on volume and watchdog
            if self.avg_db_level > self.silence_threshold and not is_watchdog_timeout:
                # We have sound
                self.silence_counter = 0
                if not self.is_playing:
                    self.sound_counter += 1
                    if self.sound_counter >= self.sound_frames_needed:
                        self.is_playing = True
                        self.is_learning = True  # Start in the learning phase
                        self.last_relearn_time = time.time()  # Start the re-learning timer
                        if self.last_bpm > 0:
                            self.send_bpm_osc(self.last_bpm)
                            self.last_update_time = time.time()
                        self.last_bpm = 0.0
            else:
                # We are silent (or the watchdog timed out)
                self.sound_counter = 0
                if self.is_playing:
                    self.silence_counter += 1
                    if self.silence_counter >= self.silence_frames_needed:
                        self.is_playing = False
                        self.is_learning = False  # Reset learning phase on silence
                        self.learning_phase_beats.clear()  # CRITICAL: Reset the learning beats
                        self.bpm_history.clear()
                        self.sanity_check_beats = 0  # Reset sanity check on silence
                        self.sanity_check_start_time = 0.0
                        self.bpm_history_raw.clear()
                        self.listening_start_time = 0.0  # Reset listening timer
                        self.send_bpm_osc(0.0)

            # --- Processing and Printing Logic ---
            if self.is_playing:
                # Check if it's time to trigger a periodic re-learn
                if self.relearn_interval > 0 and not self.is_learning and (
                        time.time() - self.last_relearn_time) > self.relearn_interval:
                    self.is_learning = True
                    self.listening_start_time = time.time()  # Start the re-learning timeout clock
                    self.learning_phase_beats.clear()
                    print(f"Re-learning BPM... (current: {self.last_bpm:.1f})                         \r")
                    # sys.stdout.write(f"Re-learning BPM... (current: {self.last_bpm:.1f})\r")

                # If we don't have a stable BPM yet, we are in a "Listening" state
                if self.is_learning:
                    # If this is the first time we are listening, start the timer
                    if self.listening_start_time == 0.0:
                        self.listening_start_time = time.time()
                    # Check for listening timeout
                    if (time.time() - self.listening_start_time) > self.listening_timeout:
                        # We've been listening for too long without finding a beat, assume silence
                        self.is_learning = False  # Exit learning phase
                        self.listening_start_time = 0.0
                        # If we were re-learning, we keep the last BPM. If we were starting from scratch, we go silent.
                        if self.last_bpm < 40.0:
                            self.is_playing = False
                            self.bpm_history.clear()
                            self.sanity_check_beats = 0
                            self.sanity_check_start_time = 0.0
                            self.bpm_history_raw.clear()
                            self.send_bpm_osc(0.0)
                            print(f"Listening timed out... Reverting to silent.                              \r")
                        else:
                            print(f"Re-learning timed out. Reverting to last BPM: {self.last_bpm:.1f}              \r")
                        # sys.stdout.write(f"Listening timed out... Reverting to silent.\r")
                    else:
                        # During the initial learning phase, we print "Listening..."
                        if self.last_bpm < 40.0:
                            sys.stdout.write(f"Listening... | Level: {self.avg_db_level:.1f} dB           \r")

                beat = self.tempo(audio_samples)
                if beat[0]:
                    current_confidence = self.tempo.get_confidence()

                    # Only process beats that meet the general confidence threshold
                    if current_confidence > self.confidence_threshold:
                        detected_bpm = self.tempo.get_bpm()

                        # Start the sanity check timer on the first confident beat of a new window
                        if self.sanity_check_start_time == 0.0:
                            self.sanity_check_start_time = time.time()
                        self.sanity_check_beats += 1

                        self.last_raw_bpm = detected_bpm  # Store the raw value at the moment of detection
                        new_bpm = detected_bpm

                        # If not in raw mode, apply the intelligent heuristics
                        if not self.raw_bpm_mode and self.is_learning:
                            # --- Learning Phase ---
                            self.learning_phase_beats.append(detected_bpm)
                            if len(self.learning_phase_beats) >= self.learning_beats_needed:
                                # We have enough beats, calculate the hypothetic BPM
                                median_bpm = np.median(self.learning_phase_beats)

                                # Anti demi-tempo robuste
                                if median_bpm < self.doubling_threshold:
                                    new_bpm = median_bpm * 2
                                else:
                                    new_bpm = median_bpm

                                # Exit learning phase and seed the history
                                self.is_learning = False
                                self.bpm_history.clear()
                                self.bpm_history_raw.clear()
                                self.bpm_history.append(new_bpm)
                                self.bpm_history_raw.append(detected_bpm)
                                self.last_relearn_time = time.time()  # Reset timer after learning is complete
                                self.last_bpm = new_bpm  # Immediately set the BPM to the corrected median

                        elif not self.raw_bpm_mode and len(self.bpm_history) > 1:
                            # --- Stable Phase: Use history for correction ---
                            print('stable phase')
                            recent_avg = np.mean(list(self.bpm_history))
                            candidates = [detected_bpm, detected_bpm * 2, detected_bpm / 2]
                            new_bpm = min(candidates, key=lambda c: abs(c - recent_avg) * (
                                1.0 if abs(c - detected_bpm) < 1 else 1.5))

                            recent_avg_raw = np.mean(list(self.bpm_history_raw))

                            # Anti demi-tempo robuste
                            if recent_avg_raw < self.doubling_threshold:
                                new_bpm = recent_avg_raw * 2

                        # --- Sanity Check Override ---
                        # This runs after the main heuristics to catch persistent half-time errors.
                        time_since_check_start = time.time() - self.sanity_check_start_time
                        if self.sanity_check_beats >= self.sanity_check_beats_needed and time_since_check_start > 0:
                            # Calculate real-world BPM based on beat frequency
                            real_world_bpm = (self.sanity_check_beats / time_since_check_start) * 60
                            # print(real_world_bpm)

                            # If the real-world BPM is roughly double our locked BPM, we have a half-time error
                            if self.last_bpm > 40 and abs((real_world_bpm / 2) - self.last_bpm) < 15:  # Use a generous threshold
                                print(f"\n[Sanity Check] Half-time error detected! Correcting {self.last_bpm:.1f} -> {self.last_bpm * 2:.1f}")
                                # Force a correction and reset history to re-stabilize at the new tempo
                                new_bpm = self.last_bpm * 2
                                self.bpm_history.clear()
                                self.bpm_history_raw.clear()

                            # Reset the sanity check for the next window
                            self.sanity_check_beats = 0
                            self.sanity_check_start_time = 0.0

                        # Also reset if the window has been open for too long (e.g., on very slow music)
                        elif self.sanity_check_start_time > 0 and time_since_check_start > self.sanity_check_window:
                            self.sanity_check_beats = 0
                            self.sanity_check_start_time = 0.0

                        if new_bpm > 0:
                            self.bpm_history.append(new_bpm)
                            self.bpm_history_raw.append(detected_bpm)

                        if self.raw_bpm_mode:
                            self.last_bpm = new_bpm
                        else:
                            self.last_bpm = (self.last_bpm * (1 - self.bpm_smoothing_factor)) + (
                                        new_bpm * self.bpm_smoothing_factor)

                        # CRITICAL: Send OSC message and update printout on every confident beat.
                        self.send_bpm_osc(self.last_bpm)
                        self.last_update_time = time.time()

                    if self.last_bpm > 0:  # This check is now just for printing
                        spinner_char = self.spinner.get_char()
                        sys.stdout.write(
                            f"{spinner_char} BPM: {self.last_bpm:.1f} | Level: {self.avg_db_level:.1f} dB  | {self.last_raw_bpm:.1f}  | {self.tempo.get_bpm():.1f}  | {current_confidence:.1f} \r")

                # Send periodic "keep-alive" updates.
                # CRITICAL: This now runs even during the re-learning phase to ensure a continuous BPM stream.
                elif self.last_bpm > 0 and (time.time() - self.last_update_time) > self.update_interval:
                    self.send_bpm_osc(self.last_bpm)
                    self.last_update_time = time.time()

            else:
                if self.last_bpm != 0.0:
                    self.last_bpm = 0.0
                sys.stdout.write(f"Silent...                                                                 \r")

            # No need for flush, stdout is line-buffered by default
        except Exception as e:
            # Catch any exception from aubio or other processing to prevent the callback thread from crashing
            print(f"\nError in audio callback: {e}", file=sys.stderr)

        return None, pyaudio.paContinue

    def send_bpm_osc(self, bpm: float):
        if not self.osc_servers:
            return

        bpmh = bpm / 2
        bpmg = math.sqrt(bpm / 240) * 100 if bpm > 0 else 0.0

        for server, s_info in zip(self.osc_servers, self.server_info):
            mode = s_info.mode or 'plain'
            value_to_send = {'plain': bpm, 'half': bpmh, 'gma3': bpmg}.get(mode.lower(), bpm)
            server[0].send_message(server[1], value_to_send)

    def stop(self):
        print("\nStopping stream and cleaning up...")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def send_bpm_osc(bpm: float, osc_servers: List[Tuple[SimpleUDPClient, str]], server_info: List[ServerInfo]):
    """Calculates different BPM modes and sends them to the configured OSC servers."""
    if not osc_servers:
        return

    # recalculate half BPM
    bpmh = bpm / 2
    # recalculate BPM for GrandMA3
    bpmg = math.sqrt(bpm / 240) * 100 if bpm > 0 else 0.0

    for server, s_info in zip(osc_servers, server_info):
        mode = s_info.mode or 'plain'
        value_to_send = {'plain': bpm, 'half': bpmh, 'gma3': bpmg}.get(mode.lower(), bpm)
        server[0].send_message(server[1], value_to_send)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Realtime Audio Beat Detector with OSC output.")
    parser.add_argument("-d", "--device", type=int, default=None,
                        help="Index of the audio input device to use (optional).")
    parser.add_argument("-st", "--silence_threshold", type=float, default=-60.0,
                        help="The volume threshold in negative dB to consider as silence (default: -60.0).")
    parser.add_argument("-c", "--confidence", type=float, default=0.2,
                        help="The confidence threshold for beat detection (0.0 to 1.0, default: 0.2).")
    parser.add_argument("--double-confidence", type=float, default=0.5,
                        help="The confidence threshold required to trigger the half-time doubling heuristic (default: 0.5).")
    parser.add_argument("-dt", "--doubling-threshold", type=float, default=100.0,
                        help="The BPM threshold below which the script will consider doubling the tempo (default: 100.0).")
    # Increased default buffer size for better accuracy on complex audio
    parser.add_argument("-b", "--bufsize", type=int, default=1024,
                        help="Size of the audio buffer for analysis (powers of 2 are best, e.g., 1024, 2048). Default: 1024.")
    parser.add_argument("--relearn-interval", type=int, default=0,
                        help="Periodically re-enter the learning phase every X seconds to adapt to tempo changes. 0 to disable (default: 0).")
    parser.add_argument("--raw-bpm", action="store_true",
                        help="Use raw BPM value from detector, bypassing intelligent heuristics.")
    parser.add_argument("-s", "--server",
                        help="OSC Server address (multiple can be provided) in format 'IP' 'PORT' 'PATH' 'MODE', "
                             "Mode PLAIN for plain BPM-Value,Mode HALF for half of BPM-Value, "
                             "Mode GMA3 for GrandMA3 Speed masters where 100 percent is for 240BPM. "
                             "  MODE is optional and default to PLAIN"
                             " e.g. 127.0.0.1 8080 /test GMA3",
                        nargs='*',
                        action="append"
                        )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command to run the meter
    run_parser = subparsers.add_parser("run", help="Run the beat detector.")

    # Command to list devices
    list_parser = subparsers.add_parser("list", help="List available audio input devices.")

    args, unknown = parser.parse_known_args()

    if args.command == "list":
        list_devices(pyaudio.PyAudio())
    else:
        # --- Device Selection ---
        p_temp = pyaudio.PyAudio()
        try:
            if args.device is not None:
                device_info = p_temp.get_device_info_by_index(args.device)
                print(f"Attempting to use specified device: [{device_info['index']}] {device_info['name']}")
            else:
                device_info = p_temp.get_default_input_device_info()
                args.device = device_info['index']
                print(f"No device specified, using default input: [{device_info['index']}] {device_info['name']}")
        except (IOError, IndexError):
            print(f"Error: Device index {args.device} is invalid. Use 'list' command to see available devices.")
            sys.exit(1)
        finally:
            p_temp.terminate()

        # --- Validate Arguments ---
        # Check if bufsize is a power of 2, which is optimal for FFT
        if not (args.bufsize > 0 and (args.bufsize & (args.bufsize - 1) == 0)):
            print(f"Warning: Buffer size {args.bufsize} is not a power of 2. This may affect performance/accuracy.",
                  file=sys.stderr)

        # --- Server Info Parsing ---
        server_info = []
        if args.server:
            server_info_4: List[ServerInfo] = [ServerInfo(x[0], int(x[1]), x[2], x[3]) for x in args.server if
                                               len(x) == 4]
            server_info_3: List[ServerInfo] = [ServerInfo(x[0], int(x[1]), x[2]) for x in args.server if len(x) == 3]
            server_info = server_info_3 + server_info_4
            for x in args.server:
                if len(x) < 3:
                    parser.error('At least 3 server arguments are required ("IP","PORT","PATH")')
                elif len(x) > 4:
                    parser.error('More than 4 arguments provided for server')
                try:
                    ipaddress.ip_address(x[0])
                except ValueError:
                    parser.error(f'Not a valid IP address: {x[0]}')
                if not x[2].startswith('/'): parser.error(f'PATH {x[2]} not valid, need to start with "/"')

        # --- Main Execution ---
        print("Starting beat detector... Press Ctrl+C to stop.")
        detector = BeatDetector(device_index=args.device, silence_threshold=args.silence_threshold,
                                server_info=server_info, confidence_threshold=args.confidence,
                                doubling_confidence_threshold=args.double_confidence, buf_size=args.bufsize,
                                raw_bpm_mode=args.raw_bpm, relearn_interval=args.relearn_interval,
                                doubling_threshold=args.doubling_threshold)

        # Set up the hotkey for manual re-learning
        if sys.platform != 'darwin':
            keyboard.add_hotkey('u', detector.trigger_relearn)
            print("Press 'u' at any time to manually trigger BPM re-learning.")

        # Keep the main thread alive until Ctrl+C
        def signal_handler(signum, frame):
            detector.stop()
            sys.exit(0)


        signal.signal(signal.SIGINT, signal_handler)

        # Use a sleep loop on Windows, as signal.pause() is not available
        if os.name == 'nt':
            while True:
                time.sleep(1)  # Main thread just sleeps, all work is in the callback
        else:
            signal.pause()