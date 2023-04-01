# openWakeWord C++

C++ version of [openWakeWord](https://github.com/dscripka/openWakeWord).

## Build

1. Download a release of the [onnxruntime](https://github.com/microsoft/onnxruntime) and extract to `lib/<arch>` where `<arch>` is `uname -m`.
2. Run `make`


## Run

After building, run:

``` sh
arecord -r 16000 -c 1 -f S16_LE -t raw - | \
  build/openwakeword --model models/alexa_v0.1.onnx
```

You can add multiple `--model <path>` arguments. See `--help` for more options.
