# openWakeWord C++

C++ version of [openWakeWord](https://github.com/dscripka/openWakeWord).

# Linux

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

# Android

- openWakeWord-cpp path: `android/app/src/main/cpp/openWakeWord-cpp`
- [onnxruntime-android](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android) path: `android/app/src/main/cpp/onnxruntime-android`
- models path moved from android/app/src/main/cpp/openWakeWord-cpp/models to: `android/app/src/main/assets/models`

android/app/build.gradle:
```gradle
android {
    defaultConfig {
        minSdkVersion 27
        ndk {
            ldLibs "log"
        }
    }
    externalNativeBuild {
        cmake {
            path "src/main/cpp/openWakeWord-cpp/src/android/CMakeLists.txt"
        }
    }
```

Example of Android service is in `openWakeWord-cpp/src/android/OpenWakeWordServiceExample.java`. There C++ part is defaulted end after waking, and started after manualy calling the service (intend) again.
- Extras with `end` property end service.
- Extras with `stop` property end cpp subprocess.
- Extras with `keyword` property start service, or only cpp subprocess, and set wake word model path. Default is `models/alexa_v0.1.onnx`.
    - Optional extras `sensitivity` as string value. Default is `0.5`.
    - Optional extras `closeServiceAfterWakeWordActivation` property end android service after waking (cpp subprocess is end by hardcode). Default is `false`.
- Don't forget to create first `NotificationChannel` in MainActivity.
- Android destroy service automaticly after same time, that's why you must set `Worker`, which will call this service each 16 minutes.
