# Onset Beat Detection to OSC

This is a simple beat detector built with [aubio](https://github.com/aubio/aubio).
It will detect the beat and BPM on the default audio input.
On every beat, the current BPM is sent to one or more OSC servers.

Main purpose of this repository is to create binary from :  https://github.com/DrLuke/aubio-beat-osc. 
This should help on the installation and setup of [WLEDAudioSync Chataigne Module](https://github.com/zak-45/WLEDAudioSync-Chataigne-Module).

## Installation

Win / Mac / Linux
```

```

Other OS (with Python installed)

see : https://github.com/DrLuke/aubio-beat-osc

## Usage

```
aubio-beat-osc beat [-h] -c IP PORT ADDRESS [-b BUFSIZE] [-v] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -c IP PORT ADDRESS, --client IP PORT ADDRESS
                        OSC Client address (multiple can be provided)
  -b BUFSIZE, --bufsize BUFSIZE
                        Size of audio buffer for beat detection (default: 128)
  -v, --verbose         Print BPM on beat
  -d DEVICE, --device DEVICE
                        Input device index (use list command to see available devices)

```

### `-c`/`--client`
Add an `IP`, `PORT` and OSC `ADDRESS` to which the BPM beat signal will be sent to. Example: `-c 127.0.0.1 31337 /foo/beat`

### `-b`/`--bufsize`
Select the size of the buffer used for beat detection.
A larger buffer is more accurate, but also more sluggish.
Refer to the [aubio](https://github.com/aubio/aubio) documentation of the tempo module for more details.
Example: `-b 128`

### `-v`/`--verbose`
Output a handy beat indicator and the current BPM to stdout.

### `-d`/`--device`
Specify the index of input device to be used.
If not provided, the default system input is used.  
Run `aubio-beat-osc list` to get all available devices.

## Example

```
$ aubio-beat-osc beat -c 127.0.0.1 12000 /WLEDAudioSync/BPM -c 10.10.13.37 12345 /test/baz -v
```

This will send beat messages to the OSC address `/WLEDAudioSync/BPM ` on `127.0.0.1:12000` and `/test/baz` on `10.10.13.37:12345`.
Additionally the current BPM will be printed to stdout.
