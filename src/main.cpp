#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace filesystem;

const string instanceName = "openWakeWord";
const size_t chunkSamples = 1280; // 80 ms
const size_t numMels = 32;
const size_t embWindowSize = 76; // 775 ms
const size_t embStepSize = 8;    // 80 ms
const size_t embFeatures = 96;
const size_t wwFeatures = 16;

void ensureArg(int argc, char *argv[], int argi);
void printUsage(char *argv[]);

struct Settings {
  path melModelPath = path("models/melspectrogram.onnx");
  path embModelPath = path("models/embedding_model.onnx");
  vector<path> wwModelPaths;

  size_t frameSize = 4 * chunkSamples;
  size_t stepFrames = 4;

  float threshold = 0.5f;
  int triggerLevel = 4;
  int refractory = 20;

  bool debug = false;

  Ort::SessionOptions options;
};

struct State {

  Ort::Env env;
  vector<mutex> mutFeatures;
  vector<condition_variable> cvFeatures;
  vector<bool> featuresExhausted;
  vector<bool> featuresReady;
  size_t numReady;
  bool samplesExhausted = false, melsExhausted = false;
  bool samplesReady = false, melsReady = false;
  mutex mutSamples, mutMels, mutReady, mutOutput;
  condition_variable cvSamples, cvMels, cvReady;

  State(size_t numWakeWords)
      : mutFeatures(numWakeWords), cvFeatures(numWakeWords),
        featuresExhausted(numWakeWords), featuresReady(numWakeWords),
        numReady(0), samplesExhausted(false), melsExhausted(false),
        samplesReady(false), melsReady(false) {
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                   instanceName.c_str());
    env.DisableTelemetryEvents();

    fill(featuresExhausted.begin(), featuresExhausted.end(), false);
    fill(featuresReady.begin(), featuresReady.end(), false);
  }
};

void audioToMels(Settings &settings, State &state, vector<float> &samplesIn,
                 vector<float> &melsOut) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  auto melSession =
      Ort::Session(state.env, settings.melModelPath.c_str(), settings.options);

  vector<int64_t> samplesShape{1, (int64_t)settings.frameSize};

  auto melInputName = melSession.GetInputNameAllocated(0, allocator);
  vector<const char *> melInputNames{melInputName.get()};

  auto melOutputName = melSession.GetOutputNameAllocated(0, allocator);
  vector<const char *> melOutputNames{melOutputName.get()};

  vector<float> todoSamples;

  {
    unique_lock lockReady(state.mutReady);
    cerr << "[LOG] Loaded mel spectrogram model" << endl;
    state.numReady += 1;
    state.cvReady.notify_one();
  }

  while (true) {
    {
      unique_lock lockSamples{state.mutSamples};
      state.cvSamples.wait(lockSamples,
                           [&state] { return state.samplesReady; });
      if (state.samplesExhausted && samplesIn.empty()) {
        break;
      }
      copy(samplesIn.begin(), samplesIn.end(), back_inserter(todoSamples));
      samplesIn.clear();

      if (!state.samplesExhausted) {
        state.samplesReady = false;
      }
    }

    while (todoSamples.size() >= settings.frameSize) {
      // Generate mels for audio samples
      vector<Ort::Value> melInputTensors;
      melInputTensors.push_back(Ort::Value::CreateTensor<float>(
          memoryInfo, todoSamples.data(), settings.frameSize,
          samplesShape.data(), samplesShape.size()));

      auto melOutputTensors =
          melSession.Run(Ort::RunOptions{nullptr}, melInputNames.data(),
                         melInputTensors.data(), melInputNames.size(),
                         melOutputNames.data(), melOutputNames.size());

      // (1, 1, frames, mels = 32)
      const auto &melOut = melOutputTensors.front();
      const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
      const auto melShape = melInfo.GetShape();

      const float *melData = melOut.GetTensorData<float>();
      size_t melCount =
          accumulate(melShape.begin(), melShape.end(), 1, multiplies<>());

      {
        unique_lock lockMels{state.mutMels};
        for (size_t i = 0; i < melCount; i++) {
          // Scale mels for Google speech embedding model
          melsOut.push_back((melData[i] / 10.0f) + 2.0f);
        }
        state.melsReady = true;
        state.cvMels.notify_one();
      }

      todoSamples.erase(todoSamples.begin(),
                        todoSamples.begin() + settings.frameSize);
    }
  }

} // audioToMels

void melsToFeatures(Settings &settings, State &state, vector<float> &melsIn,
                    vector<vector<float>> &featuresOut) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  auto embSession =
      Ort::Session(state.env, settings.embModelPath.c_str(), settings.options);

  vector<int64_t> embShape{1, (int64_t)embWindowSize, (int64_t)numMels, 1};

  auto embInputName = embSession.GetInputNameAllocated(0, allocator);
  vector<const char *> embInputNames{embInputName.get()};

  auto embOutputName = embSession.GetOutputNameAllocated(0, allocator);
  vector<const char *> embOutputNames{embOutputName.get()};

  vector<float> todoMels;
  size_t melFrames = 0;

  {
    unique_lock lockReady(state.mutReady);
    cerr << "[LOG] Loaded speech embedding model" << endl;
    state.numReady += 1;
    state.cvReady.notify_one();
  }

  while (true) {
    {
      unique_lock lockMels{state.mutMels};
      state.cvMels.wait(lockMels, [&state] { return state.melsReady; });
      if (state.melsExhausted && melsIn.empty()) {
        break;
      }
      copy(melsIn.begin(), melsIn.end(), back_inserter(todoMels));
      melsIn.clear();

      if (!state.melsExhausted) {
        state.melsReady = false;
      }
    }

    melFrames = todoMels.size() / numMels;
    while (melFrames >= embWindowSize) {
      // Generate embeddings for mels
      vector<Ort::Value> embInputTensors;
      embInputTensors.push_back(Ort::Value::CreateTensor<float>(
          memoryInfo, todoMels.data(), embWindowSize * numMels, embShape.data(),
          embShape.size()));

      auto embOutputTensors =
          embSession.Run(Ort::RunOptions{nullptr}, embInputNames.data(),
                         embInputTensors.data(), embInputTensors.size(),
                         embOutputNames.data(), embOutputNames.size());

      const auto &embOut = embOutputTensors.front();
      const auto embOutInfo = embOut.GetTensorTypeAndShapeInfo();
      const auto embOutShape = embOutInfo.GetShape();

      const float *embOutData = embOut.GetTensorData<float>();
      size_t embOutCount =
          accumulate(embOutShape.begin(), embOutShape.end(), 1, multiplies<>());

      // Send to each wake word model
      for (size_t i = 0; i < featuresOut.size(); i++) {
        unique_lock lockFeatures{state.mutFeatures[i]};
        copy(embOutData, embOutData + embOutCount,
             back_inserter(featuresOut[i]));
        state.featuresReady[i] = true;
        state.cvFeatures[i].notify_one();
      }

      // Erase a step's worth of mels
      todoMels.erase(todoMels.begin(),
                     todoMels.begin() + (embStepSize * numMels));

      melFrames = todoMels.size() / numMels;
    }
  }

} // melsToFeatures

void featuresToOutput(Settings &settings, State &state, size_t wwIdx,
                      vector<vector<float>> &featuresIn) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  auto wwModelPath = settings.wwModelPaths[wwIdx];
  auto wwName = wwModelPath.stem();
  auto wwSession =
      Ort::Session(state.env, wwModelPath.c_str(), settings.options);

  vector<int64_t> wwShape{1, (int64_t)wwFeatures, (int64_t)embFeatures};

  auto wwInputName = wwSession.GetInputNameAllocated(0, allocator);
  vector<const char *> wwInputNames{wwInputName.get()};

  auto wwOutputName = wwSession.GetOutputNameAllocated(0, allocator);
  vector<const char *> wwOutputNames{wwOutputName.get()};

  vector<float> todoFeatures;
  size_t numBufferedFeatures = 0;
  int activation = 0;

  {
    unique_lock lockReady(state.mutReady);
    cerr << "[LOG] Loaded " << wwName << " model" << endl;
    state.numReady += 1;
    state.cvReady.notify_one();
  }

  while (true) {
    {
      unique_lock lockFeatures{state.mutFeatures[wwIdx]};
      state.cvFeatures[wwIdx].wait(
          lockFeatures, [&state, wwIdx] { return state.featuresReady[wwIdx]; });
      if (state.featuresExhausted[wwIdx] && featuresIn[wwIdx].empty()) {
        break;
      }
      copy(featuresIn[wwIdx].begin(), featuresIn[wwIdx].end(),
           back_inserter(todoFeatures));
      featuresIn[wwIdx].clear();

      if (!state.featuresExhausted[wwIdx]) {
        state.featuresReady[wwIdx] = false;
      }
    }

    numBufferedFeatures = todoFeatures.size() / embFeatures;
    while (numBufferedFeatures >= wwFeatures) {
      vector<Ort::Value> wwInputTensors;
      wwInputTensors.push_back(Ort::Value::CreateTensor<float>(
          memoryInfo, todoFeatures.data(), wwFeatures * embFeatures,
          wwShape.data(), wwShape.size()));

      auto wwOutputTensors =
          wwSession.Run(Ort::RunOptions{nullptr}, wwInputNames.data(),
                        wwInputTensors.data(), 1, wwOutputNames.data(), 1);

      const auto &wwOut = wwOutputTensors.front();
      const auto wwOutInfo = wwOut.GetTensorTypeAndShapeInfo();
      const auto wwOutShape = wwOutInfo.GetShape();
      const float *wwOutData = wwOut.GetTensorData<float>();
      size_t wwOutCount =
          accumulate(wwOutShape.begin(), wwOutShape.end(), 1, multiplies<>());

      for (size_t i = 0; i < wwOutCount; i++) {
        auto probability = wwOutData[i];
        if (settings.debug) {
          {
            unique_lock lockOutput(state.mutOutput);
            cerr << wwName << " " << probability << endl;
          }
        }

        if (probability > settings.threshold) {
          // Activated
          activation++;
          if (activation >= settings.triggerLevel) {
            // Trigger level reached
            {
              unique_lock lockOutput(state.mutOutput);
              cout << wwName << endl;
            }
            activation = -settings.refractory;
          }
        } else {
          // Back towards 0
          if (activation > 0) {
            activation = max(0, activation - 1);
          } else {
            activation = min(0, activation + 1);
          }
        }
      }

      // Remove 1 embedding
      todoFeatures.erase(todoFeatures.begin(),
                         todoFeatures.begin() + (1 * embFeatures));

      numBufferedFeatures = todoFeatures.size() / embFeatures;
    }
  }

} // featuresToOutput

int main(int argc, char *argv[]) {

  // Re-open stdin/stdout in binary mode
  freopen(NULL, "rb", stdin);

  Settings settings;

  // Parse arguments
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];

    if (arg == "-m" || arg == "--model") {
      ensureArg(argc, argv, i);
      settings.wwModelPaths.push_back(path(argv[++i]));
    } else if (arg == "-t" || arg == "--threshold") {
      ensureArg(argc, argv, i);
      settings.threshold = atof(argv[++i]);
    } else if (arg == "-l" || arg == "--trigger-level") {
      ensureArg(argc, argv, i);
      settings.triggerLevel = atoi(argv[++i]);
    } else if (arg == "-r" || arg == "--refractory") {
      ensureArg(argc, argv, i);
      settings.refractory = atoi(argv[++i]);
    } else if (arg == "--step-frames") {
      ensureArg(argc, argv, i);
      settings.stepFrames = atoi(argv[++i]);
    } else if (arg == "--melspectrogram-model") {
      ensureArg(argc, argv, i);
      settings.melModelPath = path(argv[++i]);
    } else if (arg == "--embedding-model") {
      ensureArg(argc, argv, i);
      settings.embModelPath = path(argv[++i]);
    } else if (arg == "--debug") {
      settings.debug = true;
    } else if (arg == "-h" || arg == "--help") {
      printUsage(argv);
      exit(0);
    }
  }

  if (settings.wwModelPaths.empty()) {
    cerr << "[ERROR] --model is required" << endl;
    return 1;
  }

  settings.frameSize = settings.stepFrames * chunkSamples;

  // Absolutely critical for performance
  settings.options.SetIntraOpNumThreads(1);
  settings.options.SetInterOpNumThreads(1);

  const size_t numWakeWords = settings.wwModelPaths.size();
  State state(numWakeWords);

  vector<float> floatSamples;
  vector<float> mels;
  vector<vector<float>> features(numWakeWords);

  thread melThread(audioToMels, ref(settings), ref(state), ref(floatSamples),
                   ref(mels));
  thread featuresThread(melsToFeatures, ref(settings), ref(state), ref(mels),
                        ref(features));

  vector<thread> wwThreads;
  for (size_t i = 0; i < numWakeWords; i++) {
    wwThreads.push_back(
        thread(featuresToOutput, ref(settings), ref(state), i, ref(features)));
  }

  // Block until ready
  const size_t numReadyExpected = 2 + numWakeWords;
  {
    unique_lock lockReady(state.mutReady);
    state.cvReady.wait(lockReady, [&state, numReadyExpected] {
      return state.numReady == numReadyExpected;
    });
  }

  cerr << "[LOG] Ready" << endl;

  // Main loop
  int16_t samples[settings.frameSize];
  size_t framesRead =
      fread(samples, sizeof(int16_t), settings.frameSize, stdin);

  while (framesRead > 0) {
    {
      unique_lock lockSamples{state.mutSamples};

      for (size_t i = 0; i < framesRead; i++) {
        // NOTE: we do NOT normalize here
        floatSamples.push_back((float)samples[i]);
      }

      state.samplesReady = true;
      state.cvSamples.notify_one();
    }

    // Next samples
    framesRead = fread(samples, sizeof(int16_t), settings.frameSize, stdin);
  }

  // Signal mel thread that samples have been exhausted
  {
    unique_lock lockSamples{state.mutSamples};
    state.samplesExhausted = true;
    state.samplesReady = true;
    state.cvSamples.notify_one();
  }

  melThread.join();

  // Signal features thread that mels have been exhausted
  {
    unique_lock lockMels{state.mutMels};
    state.melsExhausted = true;
    state.melsReady = true;
    state.cvMels.notify_one();
  }
  featuresThread.join();

  // Signal wake word threads that features have been exhausted
  for (size_t i = 0; i < numWakeWords; i++) {
    unique_lock lockFeatures{state.mutFeatures[i]};
    state.featuresExhausted[i] = true;
    state.featuresReady[i] = true;
    state.cvFeatures[i].notify_one();
  }

  for (size_t i = 0; i < numWakeWords; i++) {
    wwThreads[i].join();
  }

  return 0;
}

void printUsage(char *argv[]) {
  cerr << endl;
  cerr << "usage: " << argv[0] << " [options]" << endl;
  cerr << endl;
  cerr << "options:" << endl;
  cerr << "   -h        --help                  show this message and exit"
       << endl;
  cerr << "   -m  FILE  --model          FILE   path to wake word model "
          "(repeat "
          "for multiple models)"
       << endl;
  cerr << "   -t  NUM   --threshold      NUM    threshold for activation (0-1, "
          "default: 0.5)"
       << endl;
  cerr << "   -l  NUM   --trigger-level  NUM    number of activations before "
          "output (default: 4)"
       << endl;
  cerr << "   -r  NUM   --refractory     NUM    number of steps after "
          "activation to wait (default: 20)"
       << endl;
  cerr
      << "   --step-frames              NUM    number of 80 ms audio chunks to "
         "process at a time (default: 4)"
      << endl;
  cerr << "   --melspectrogram-model     FILE   path to "
          "melspectrogram.onnx file"
       << endl;
  cerr << "   --embedding-model          FILE   path to "
          "embedding_model.onnx file"
       << endl;
  cerr << "   --debug                           print model probabilities to "
          "stderr"
       << endl;
  cerr << endl;
}

void ensureArg(int argc, char *argv[], int argi) {
  if ((argi + 1) >= argc) {
    printUsage(argv);
    exit(0);
  }
}
