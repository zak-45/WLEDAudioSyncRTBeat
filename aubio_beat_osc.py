import pyaudio
import numpy as np
import aubio
import signal
import os
import time

import argparse

from pythonosc.udp_client import SimpleUDPClient

from typing import List, NamedTuple, Tuple


class ClientInfo(NamedTuple):
    ip: str
    port: int
    address: str


parser = argparse.ArgumentParser()
sp = parser.add_subparsers(dest="command")

beat_parser = sp.add_parser("beat", help="Start beat detection")
beat_parser.add_argument("-s", "--server", help="OSC server address (multiple can be provided)", nargs=3,
                         action="append",
                         metavar=("IP", "PORT", "ADDRESS"), required=True)
beat_parser.add_argument("-b", "--bufsize", help="Size of audio buffer for beat detection (default: 128)", default=128,
                         type=int)
beat_parser.add_argument("-v", "--verbose", help="Print BPM on beat", action="store_true")
beat_parser.add_argument("-d", "--device", help="Input device index (use list command to see available devices)",
                         default=None, type=int)

list_parser = sp.add_parser("list", help="Print a list of all available audio devices")
args = parser.parse_args()


class BeatPrinter:
    def __init__(self):
        self.state: int = 0
        self.spinner = "▚▞"

    def print_bpm(self, bpm: float) -> None:
        print(f"{self.spinner[self.state]}\t{bpm:.1f} BPM")
        self.state = (self.state + 1) % len(self.spinner)


class BeatDetector:
    def __init__(self, buf_size: int, client_infos: List[ClientInfo]):
        self.buf_size: int = buf_size
        self.client_infos: List[ClientInfo] = client_infos

        # Set up pyaudio and aubio beat detector
        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        samplerate: int = 44100
        self.stream: pyaudio.Stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=samplerate,
            input=True,
            frames_per_buffer=self.buf_size,
            stream_callback=self._pyaudio_callback,
            input_device_index=args.device
        )

        fft_size: int = self.buf_size * 2
        self.tempo: aubio.tempo = aubio.tempo("default", fft_size, self.buf_size, samplerate)

        # Set up OSC servers to send beat data to
        self.osc_clients: List[Tuple[SimpleUDPClient, str]] = [(SimpleUDPClient(x.ip, x.port), x.address) for x in
                                                               self.client_infos]

        self.spinner: BeatPrinter = BeatPrinter()

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        signal = np.frombuffer(in_data, dtype=np.float32)
        beat = self.tempo(signal)
        if beat[0]:
            if args.verbose:
                self.spinner.print_bpm(self.tempo.get_bpm())
            for client in self.osc_clients:
                client[0].send_message(client[1], self.tempo.get_bpm())

        return None, pyaudio.paContinue  # Tell pyAudio to continue

    def __del__(self):
        self.stream.close()
        self.p.terminate()


def list_devices():
    print("Listing all available input devices:\n")
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"[{i}] {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

    print("\nUse the number in the square brackets as device index")


def main():
    if args.command == "list":
        list_devices()
        return

    if args.command == "beat":
        # Pack data from arguments into ClientInfo objects
        client_infos: List[ClientInfo] = [ClientInfo(x[0], int(x[1]), x[2]) for x in args.client]

        bd = BeatDetector(args.bufsize, client_infos)

        # Audio processing happens in separate thread, so put this thread to sleep
        if os.name == 'nt':  # Windows is not able to pause the main thread :(
            while True:
                time.sleep(1)
        else:
            signal.pause()


if __name__ == "__main__":
    main()
