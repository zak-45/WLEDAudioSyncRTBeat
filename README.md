# Onset Beat Detection to OSC
[![Cross Compile Manual workflow](https://github.com/zak-45/WLEDAudioSyncRTBeat/actions/workflows/manual.yml/badge.svg)](https://github.com/zak-45/WLEDAudioSyncRTBeat/actions/workflows/manual.yml)

This is a real-time beat and tempo detector built with Python and the [aubio](https://github.com/aubio/aubio) library. It captures audio from a selected input device, analyzes it to find the rhythm, and sends the calculated Beats Per Minute (BPM) to one or more OSC (Open Sound Control) servers.

It is designed to be a stable, low-latency bridge between live audio and lighting software, VJ applications, or any other OSC-compatible system.

This is a feature of [WLEDAudioSync Chataigne Module](https://github.com/zak-45/WLEDAudioSync-Chataigne-Module).

You can see a demo here : [WLEDAudioSyncRTBeat demo](https://youtu.be/VXM_zEzKo6M)

Chataigne OSC Server view:

![image](https://github.com/zak-45/WLEDAudioSyncRTBeat/assets/121941293/89b89dbf-49bb-410e-8d7b-2c43357c5100)


## Installation

### Packaged Release (Recommended for Win / Mac / Linux)

Grab the latest pre-compiled, portable release from here: https://github.com/zak-45/WLEDAudioSyncRTBeat/releases

```
No Python installation is needed.
This is a portable version.
Just place it in a folder and run the executable for your OS.
```

**INFO**
---
Some antivirus software may flag the executable as a potential threat. This is a false positive due to the way the script is packaged into a single file. If you do not trust the executable, you can always run the script from source using the Python method below.
---

### From Source (All OS with Python)

1.  Install the required Python modules:
    ```
    pip install -r requirements.txt
    ```
    *(Note: This will install `pyaudio`, `numpy`, `aubio`, `python-osc`, and `keyboard`.)*

2.  Download the `WLEDAudioSyncRTBeat.py` file and run it:
    ```
    python WLEDAudioSyncRTBeat.py
    ```

## Usage

The script can be launched with the `run` command (which is the default) or the `list` command to see audio devices.

```
usage: WLEDAudioSyncRTBeat.py [-h] [-d DEVICE] [-st SILENCE_THRESHOLD] [-c CONFIDENCE] [--double-confidence DOUBLE_CONFIDENCE] [--doubling-threshold DOUBLING_THRESHOLD] [-b BUFSIZE] [--relearn-interval RELEARN_INTERVAL] [--raw-bpm] [-s IP PORT ADDRESS [MODE]] [{run,list}] ...

Realtime Audio Beat Detector with OSC output.
```

### Commands

`run`
:   (Default) Starts the beat detector. If no command is specified, `run` is executed automatically.

`list`
:   Lists all available audio input devices and their corresponding index numbers, which can be used with the `-d` flag.

### Options

`-h, --help`
:   Shows the help message and exits.

`-d DEVICE, --device DEVICE`
:   Specifies the index of the audio input device to use. If not provided, the system's default input device is used automatically.

`-st SILENCE_THRESHOLD, --silence_threshold SILENCE_THRESHOLD`
:   Sets the volume threshold (in negative dB) to be considered silence. Default: `-60.0`.

`-c CONFIDENCE, --confidence CONFIDENCE`
:   The general confidence threshold (0.0 to 1.0) that `aubio` must have in a beat for it to be processed. Lower values are more sensitive but can be less stable. Default: `0.2`.

`--double-confidence DOUBLE_CONFIDENCE`
:   The higher confidence threshold (0.0 to 1.0) required to trigger the more aggressive half-time doubling heuristic. Default: `0.5`.

`-dt DOUBLING_THRESHOLD, --doubling-threshold DOUBLING_THRESHOLD` 
:   The BPM threshold below which the script will consider doubling the tempo to correct for half-time errors. Default: `100.0`.

`-b BUFSIZE, --bufsize BUFSIZE`
:   The size of the audio buffer for analysis. Larger values (e.g., 2048) can improve accuracy on complex music at the cost of slightly higher latency. Powers of 2 are optimal (512, 1024, 2048). Default: `1024`.

`--relearn-interval RELEARN_INTERVAL`
:   A powerful adaptive feature. The script will periodically re-enter the "learning phase" every X seconds to adapt to tempo changes. This is ideal for long DJ sets or songs with tempo shifts. Set to `0` to disable. Default: `0`.

`--raw-bpm`
:   A debug mode that disables all the intelligent BPM correction heuristics. This shows you the raw, unfiltered tempo directly from the `aubio` detector.

`-s IP PORT ADDRESS [MODE], --server IP PORT ADDRESS [MODE]`
:   The destination for the OSC messages. This argument can be used multiple times to send data to multiple servers simultaneously.
    *   `IP`: The IP address of the OSC server.
    *   `PORT`: The port of the OSC server.
    *   `ADDRESS`: The OSC path (e.g., `/wled/bpm`).
    *   `MODE` (Optional): Can be `PLAIN` (the final BPM), `HALF` (BPM / 2), or `GMA3` (a specific curve for GrandMA3 lighting software). Defaults to `PLAIN`.

### Interactive Controls

`u` key
:   Press the `u` key at any time to manually trigger a BPM re-learning phase. This is useful if you feel the tempo is incorrect or during a song transition.

## Example

```sh
# Run the detector, sending BPM to two different servers with different modes,
# and have it re-evaluate the tempo every 90 seconds.
python WLEDAudioSyncRTBeat.py --relearn-interval 90 -s 127.0.0.1 12000 /wled/bpm -s 192.168.1.50 8000 /gma3/speed GMA3
```

This will:
1.  Start the beat detector using the default audio device.
2.  Send the calculated BPM to `/wled/bpm` on the local machine at port `12000`.
3.  Send a specially formatted BPM value to `/gma3/speed` on a device at `192.168.1.50:8000`.
4.  Every 90 seconds, it will non-disruptively re-analyze the audio to confirm or update the tempo.

## Credits

Thanks to :  https://github.com/DrLuke/aubio-beat-osc for the original inspiration.