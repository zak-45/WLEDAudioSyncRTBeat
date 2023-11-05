# Onset Beat Detection to OSC
[![Cross Compile Manual workflow](https://github.com/zak-45/WLEDAudioSyncRTBeat/actions/workflows/manual.yml/badge.svg)](https://github.com/zak-45/WLEDAudioSyncRTBeat/actions/workflows/manual.yml)

This is a simple beat detector built with [aubio](https://github.com/aubio/aubio).
It will detect the beat and BPM on the default audio input.
On every beat, the current BPM is sent to one or more OSC servers.

Main purpose of this repository is to create binary from :  https://github.com/DrLuke/aubio-beat-osc. 
This should help on the installation and setup of [WLEDAudioSync Chataigne Module](https://github.com/zak-45/WLEDAudioSync-Chataigne-Module).

## Installation

Win / Mac / Linux

Take your release from there : https://github.com/zak-45/WLEDAudioSyncRTBeat/releases

```
This is a portable version, put it on a nice folder and just run it according your OS.
```

Other OS (with Python installed)

see : https://github.com/DrLuke/aubio-beat-osc

## Usage

```
WLEDAudioSyncRTBeat-{OS} beat|list [-h] -s IP PORT ADDRESS [-b BUFSIZE] [-v] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -s IP PORT ADDRESS, --server IP PORT ADDRESS
                        OSC Server address (multiple can be provided)
  -b BUFSIZE, --bufsize BUFSIZE
                        Size of audio buffer for beat detection (default: 128)
  -v, --verbose         Print BPM on beat
  -d DEVICE, --device DEVICE
                        Input device index (use list command to see available devices)

```

### `-s`/`--server`
Add an `IP`, `PORT` and OSC `ADDRESS` to which the BPM beat signal will be sent to. Example: `-s 127.0.0.1 21000 /foo/beat`

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
Run `WLEDAudioSyncRTBeat list` to get all available devices.

## Example

```
$ WLEDAudioSyncRTBeat-Linux beat -s 127.0.0.1 12000 /WLEDAudioSync/BPM -s 10.10.13.37 12345 /test/baz -v
```

This will send beat messages to the OSC address `/WLEDAudioSync/BPM ` on `127.0.0.1:12000` and `/test/baz` on `10.10.13.37:12345`.
Additionally the current BPM will be printed to stdout.

## Info 

```
First time you run WLEDAudioSyncRTBeat-{OS},
this will create folder ./WLEDAudioSyncRTBeat and extract all files on it.

To save some space and time,
you can then delete WLEDAudioSyncRTBeat-* and run the app from created folder.
```
