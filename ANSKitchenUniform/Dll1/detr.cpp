#include "ANSKitchenUniform.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <windows.h>
#include <opencv2/dnn.hpp>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace {
    struct DetrCandidate {
        int index;
        int classId;
        float confidence;
        cv::Rect box;
    };

    cv::Mat findOutput(const std::vector<cv::Mat>& outputs,
        const std::vector<std::string>& names, const std::string& wantedName) {
        const auto found = std::find(names.begin(), names.end(), wantedName);
        if (found == names.end()) return {};
        return outputs[static_cast<size_t>(std::distance(names.begin(), found))];
    }

    void checkOrtStatus(const OrtApi* api, OrtStatus* status, const std::string& operation) {
        if (!status) return;
        const char* message = api->GetErrorMessage(status);
        const std::string details = message ? message : "unknown ONNX Runtime error";
        api->ReleaseStatus(status);
        throw std::runtime_error(operation + ": " + details);
    }

    class OrtValueCollection {
    public:
        explicit OrtValueCollection(const OrtApi* api) : _api(api) {}
        ~OrtValueCollection() {
            for (OrtValue* value : values) {
                if (value) _api->ReleaseValue(value);
            }
        }

        OrtValueCollection(const OrtValueCollection&) = delete;
        OrtValueCollection& operator=(const OrtValueCollection&) = delete;

        std::vector<OrtValue*> values;

    private:
        const OrtApi* _api;
    };

    struct OrtValueDeleter {
        const OrtApi* api;
        void operator()(OrtValue* value) const {
            if (value) api->ReleaseValue(value);
        }
    };

    cv::Mat getFloatOutput(const OrtApi* api, OrtValue* output) {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        checkOrtStatus(api, api->GetTensorTypeAndShape(output, &info),
            "Reading output tensor metadata");

        cv::Mat result;
        try {
            ONNXTensorElementDataType elementType;
            checkOrtStatus(api, api->GetTensorElementType(info, &elementType),
                "Reading output tensor element type");
            if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::runtime_error("RF-DETR output tensor is not float32");
            }

            size_t rank = 0;
            checkOrtStatus(api, api->GetDimensionsCount(info, &rank),
                "Reading output tensor rank");
            if (rank == 0 || rank > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("RF-DETR output tensor has an invalid rank");
            }
            std::vector<int64_t> ortDimensions(rank);
            checkOrtStatus(api, api->GetDimensions(info, ortDimensions.data(), rank),
                "Reading output tensor dimensions");

            std::vector<int> dimensions;
            dimensions.reserve(rank);
            for (const int64_t dimension : ortDimensions) {
                if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
                    throw std::runtime_error("RF-DETR output tensor has an invalid dimension");
                }
                dimensions.push_back(static_cast<int>(dimension));
            }

            void* data = nullptr;
            checkOrtStatus(api, api->GetTensorMutableData(output, &data),
                "Reading output tensor data");
            result = cv::Mat(static_cast<int>(dimensions.size()), dimensions.data(), CV_32F, data);
        }
        catch (...) {
            api->ReleaseTensorTypeAndShapeInfo(info);
            throw;
        }
        api->ReleaseTensorTypeAndShapeInfo(info);
        return result;
    }
}

bool DetrUniformDetector::Initialize(
    const std::string& modelPath, const std::string& classNamesPath, float scoreThreshold) {
    Destroy();
    try {
        std::ifstream namesFile(classNamesPath);
        if (!namesFile.is_open()) {
            std::cerr << "[DetrUniformDetector] Cannot open class names file: "
                << classNamesPath << std::endl;
            return false;
        }

        std::string className;
        while (std::getline(namesFile, className)) {
            const size_t first = className.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            const size_t last = className.find_last_not_of(" \t\r\n");
            _classNames.push_back(className.substr(first, last - first + 1));
        }
        if (_classNames.size() != 4) {
            throw std::runtime_error("Expected exactly four RF-DETR class names in " + classNamesPath);
        }

        wchar_t programDataBuffer[MAX_PATH]{};
        const DWORD programDataLength = GetEnvironmentVariableW(
            L"ProgramData", programDataBuffer, MAX_PATH);
        const std::filesystem::path sharedDirectory =
            (programDataLength > 0 && programDataLength < MAX_PATH
                ? std::filesystem::path(programDataBuffer)
                : std::filesystem::path(L"C:\\ProgramData")) /
            L"ANSCENTER" / L"Shared";
        const std::vector<std::filesystem::path> runtimeCandidates{
            sharedDirectory / L"ep" / L"cuda" / L"onnxruntime.dll",
            sharedDirectory / L"ep" / L"cpu" / L"onnxruntime.dll",
            sharedDirectory / L"ep" / L"directml" / L"onnxruntime.dll",
            sharedDirectory / L"onnxruntime.dll"
        };
        SetDllDirectoryW(sharedDirectory.c_str());
        for (const auto& candidate : runtimeCandidates) {
            _runtimeLibrary = LoadLibraryExW(candidate.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (_runtimeLibrary) {
                std::cout << "[DetrUniformDetector] Loaded ONNX Runtime: "
                    << candidate.string() << std::endl;
                break;
            }
        }
        if (!_runtimeLibrary) {
            throw std::runtime_error(
                "Cannot load ONNX Runtime from the ANSCENTER shared runtime directories");
        }

        const auto getApiBase = reinterpret_cast<decltype(&OrtGetApiBase)>(
            GetProcAddress(static_cast<HMODULE>(_runtimeLibrary), "OrtGetApiBase"));
        if (!getApiBase) {
            throw std::runtime_error("Loaded ONNX Runtime does not export OrtGetApiBase");
        }
        const OrtApiBase* apiBase = getApiBase();
        _api = apiBase ? apiBase->GetApi(ORT_API_VERSION) : nullptr;
        if (!_api) {
            throw std::runtime_error("ONNX Runtime does not support the C API version in the bundled header");
        }

        checkOrtStatus(_api, _api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ANSKitchenUniform", &_environment),
            "Creating ONNX Runtime environment");
        checkOrtStatus(_api, _api->CreateSessionOptions(&_sessionOptions),
            "Creating ONNX Runtime session options");
        checkOrtStatus(_api, _api->SetSessionGraphOptimizationLevel(_sessionOptions, ORT_ENABLE_ALL),
            "Enabling ONNX Runtime graph optimizations");

        OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
        OrtStatus* cudaStatus = _api->CreateCUDAProviderOptions(&cudaOptions);
        if (!cudaStatus && cudaOptions) {
            cudaStatus = _api->SessionOptionsAppendExecutionProvider_CUDA_V2(_sessionOptions, cudaOptions);
            if (!cudaStatus) {
                _usingCuda = true;
            }
            else {
                const char* message = _api->GetErrorMessage(cudaStatus);
                std::cerr << "[DetrUniformDetector] CUDA provider unavailable; using CPU: "
                    << (message ? message : "unknown error") << std::endl;
                _api->ReleaseStatus(cudaStatus);
            }
            _api->ReleaseCUDAProviderOptions(cudaOptions);
        }
        else if (cudaStatus) {
            const char* message = _api->GetErrorMessage(cudaStatus);
            std::cerr << "[DetrUniformDetector] CUDA provider unavailable; using CPU: "
                << (message ? message : "unknown error") << std::endl;
            _api->ReleaseStatus(cudaStatus);
        }

        const std::filesystem::path modelFilePath(modelPath);
        OrtStatus* sessionStatus = _api->CreateSession(
            _environment, modelFilePath.c_str(), _sessionOptions, &_session);
        if (sessionStatus && _usingCuda) {
            const char* message = _api->GetErrorMessage(sessionStatus);
            std::cerr << "[DetrUniformDetector] CUDA session creation failed; retrying on CPU: "
                << (message ? message : "unknown error") << std::endl;
            _api->ReleaseStatus(sessionStatus);
            _session = nullptr;
            _api->ReleaseSessionOptions(_sessionOptions);
            _sessionOptions = nullptr;
            _usingCuda = false;
            checkOrtStatus(_api, _api->CreateSessionOptions(&_sessionOptions),
                "Creating CPU ONNX Runtime session options");
            checkOrtStatus(_api, _api->SetSessionGraphOptimizationLevel(_sessionOptions, ORT_ENABLE_ALL),
                "Enabling CPU ONNX Runtime graph optimizations");
            sessionStatus = _api->CreateSession(
                _environment, modelFilePath.c_str(), _sessionOptions, &_session);
        }
        checkOrtStatus(_api, sessionStatus, "Loading RF-DETR ONNX model");

        size_t inputCount = 0;
        checkOrtStatus(_api, _api->SessionGetInputCount(_session, &inputCount),
            "Reading model input count");
        if (inputCount != 1) throw std::runtime_error("RF-DETR model must have exactly one input");

        OrtAllocator* allocator = nullptr;
        checkOrtStatus(_api, _api->GetAllocatorWithDefaultOptions(&allocator),
            "Getting ONNX Runtime allocator");
        char* inputName = nullptr;
        checkOrtStatus(_api, _api->SessionGetInputName(_session, 0, allocator, &inputName),
            "Reading model input name");
        _inputName = inputName;
        checkOrtStatus(_api, _api->AllocatorFree(allocator, inputName),
            "Releasing model input name");

        size_t outputCount = 0;
        checkOrtStatus(_api, _api->SessionGetOutputCount(_session, &outputCount),
            "Reading model output count");
        for (size_t index = 0; index < outputCount; ++index) {
            char* outputName = nullptr;
            checkOrtStatus(_api, _api->SessionGetOutputName(_session, index, allocator, &outputName),
                "Reading model output name");
            _outputNames.emplace_back(outputName);
            checkOrtStatus(_api, _api->AllocatorFree(allocator, outputName),
                "Releasing model output name");
        }
        for (const std::string& expected : { "dets", "labels", "masks" }) {
            if (std::find(_outputNames.begin(), _outputNames.end(), expected) == _outputNames.end()) {
                throw std::runtime_error("Required RF-DETR output is missing: " + expected);
            }
        }

        _scoreThreshold = std::clamp(scoreThreshold, 0.0f, 1.0f);
        checkOrtStatus(_api, _api->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &_memoryInfo),
            "Creating ONNX Runtime input memory info");

        std::cout << "[DetrUniformDetector] Loaded RF-DETR ONNX with ONNX Runtime: " << modelPath
            << " (input " << _inputWidth << "x" << _inputHeight
            << ", outputs dets/labels/masks, backend "
            << (_usingCuda ? "CUDA" : "CPU") << ")." << std::endl;
        return true;
    }
    catch (const std::exception& error) {
        std::cerr << "[DetrUniformDetector] Initialization failed: "
            << error.what() << std::endl;
    }
    Destroy();
    return false;
}

int DetrUniformDetector::RunInference(
    const cv::Mat& input, const std::string& cameraId, std::vector<ANSCENTER::Object>& detections) {
    detections.clear();
    if (input.empty() || !_session || !_api || !_memoryInfo) {
        std::cerr << "[DetrUniformDetector] Inference called with an empty frame or unloaded model."
            << std::endl;
        return 0;
    }

    try {
        const float scale = std::min(
            static_cast<float>(_inputWidth) / input.cols,
            static_cast<float>(_inputHeight) / input.rows);
        const int resizedWidth = std::max(1, cvRound(input.cols * scale));
        const int resizedHeight = std::max(1, cvRound(input.rows * scale));
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(resizedWidth, resizedHeight),
            0.0, 0.0, cv::INTER_LINEAR);
        const int padX = (_inputWidth - resizedWidth) / 2;
        const int padY = (_inputHeight - resizedHeight) / 2;
        cv::Mat letterboxed(_inputHeight, _inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(letterboxed(cv::Rect(padX, padY, resizedWidth, resizedHeight)));
        cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed, 1.0 / 255.0, cv::Size(_inputWidth, _inputHeight),
            cv::Scalar(), true, false, CV_32F);
        const std::array<int64_t, 4> inputDimensions{
            1, 3, _inputHeight, _inputWidth
        };
        OrtValue* rawInputValue = nullptr;
        checkOrtStatus(_api, _api->CreateTensorWithDataAsOrtValue(
            _memoryInfo, blob.ptr<float>(), blob.total() * sizeof(float),
            inputDimensions.data(), inputDimensions.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rawInputValue),
            "Creating RF-DETR input tensor");
        std::unique_ptr<OrtValue, OrtValueDeleter> inputValue(
            rawInputValue, OrtValueDeleter{ _api });

        const char* inputNames[] = { _inputName.c_str() };
        std::vector<const char*> outputNamePointers;
        outputNamePointers.reserve(_outputNames.size());
        for (const std::string& outputName : _outputNames) {
            outputNamePointers.push_back(outputName.c_str());
        }
        OrtValueCollection ortOutputs(_api);
        ortOutputs.values.resize(_outputNames.size(), nullptr);
        const OrtValue* inputValues[] = { inputValue.get() };
        checkOrtStatus(_api, _api->Run(
            _session, nullptr, inputNames, inputValues, 1,
            outputNamePointers.data(), outputNamePointers.size(), ortOutputs.values.data()),
            "Running RF-DETR inference");

        std::vector<cv::Mat> outputs;
        outputs.reserve(ortOutputs.values.size());
        for (OrtValue* output : ortOutputs.values) {
            outputs.push_back(getFloatOutput(_api, output));
        }

        const cv::Mat dets = findOutput(outputs, _outputNames, "dets");
        const cv::Mat labels = findOutput(outputs, _outputNames, "labels");
        const cv::Mat masks = findOutput(outputs, _outputNames, "masks");
        if (dets.empty() || labels.empty() || masks.empty() ||
            dets.type() != CV_32F || labels.type() != CV_32F || masks.type() != CV_32F ||
            dets.dims != 3 || labels.dims != 3 || masks.dims != 4 ||
            dets.size[0] != 1 || labels.size[0] != 1 || masks.size[0] != 1 ||
            dets.size[2] != 4 || labels.size[1] != dets.size[1] ||
            masks.size[1] != dets.size[1] || labels.size[2] != static_cast<int>(_classNames.size()) + 1 ||
            masks.size[2] <= 0 || masks.size[3] <= 0) {
            std::cerr << "[DetrUniformDetector] Unexpected RF-DETR output shapes or types."
                << std::endl;
            return 0;
        }

        const int candidateCount = dets.size[1];
        const int classLogitCount = labels.size[2];
        const int maskHeight = masks.size[2];
        const int maskWidth = masks.size[3];
        const float* boxData = dets.ptr<float>();
        const float* classData = labels.ptr<float>();
        std::vector<DetrCandidate> candidates;
        candidates.reserve(candidateCount);
        const cv::Rect frameBounds(0, 0, input.cols, input.rows);

        for (int index = 0; index < candidateCount; ++index) {
            const float* logits = classData + static_cast<size_t>(index) * classLogitCount;
            float maxLogit = -std::numeric_limits<float>::infinity();
            int classId = -1;
            for (int classIndex = 0; classIndex < static_cast<int>(_classNames.size()); ++classIndex) {
                if (std::isfinite(logits[classIndex]) && logits[classIndex] > maxLogit) {
                    maxLogit = logits[classIndex];
                    classId = classIndex;
                }
            }
            if (classId < 0 || !std::isfinite(maxLogit)) continue;

            const double clippedLogit = std::clamp(
                static_cast<double>(logits[classId]), -80.0, 80.0);
            const float confidence = static_cast<float>(
                1.0 / (1.0 + std::exp(-clippedLogit)));
            if (confidence < _scoreThreshold) continue;

            const float* box = boxData + static_cast<size_t>(index) * 4;
            if (!std::isfinite(box[0]) || !std::isfinite(box[1]) ||
                !std::isfinite(box[2]) || !std::isfinite(box[3]) ||
                box[2] <= 0.0f || box[3] <= 0.0f) continue;

            const float canvasX1 = (box[0] - box[2] * 0.5f) * _inputWidth;
            const float canvasY1 = (box[1] - box[3] * 0.5f) * _inputHeight;
            const float canvasX2 = (box[0] + box[2] * 0.5f) * _inputWidth;
            const float canvasY2 = (box[1] + box[3] * 0.5f) * _inputHeight;
            const int x1 = cvRound((canvasX1 - padX) / scale);
            const int y1 = cvRound((canvasY1 - padY) / scale);
            const int x2 = cvRound((canvasX2 - padX) / scale);
            const int y2 = cvRound((canvasY2 - padY) / scale);
            const cv::Rect clippedBox(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
            const cv::Rect frameBox = clippedBox & frameBounds;
            if (frameBox.area() <= 0) continue;
            candidates.push_back({ index, classId, confidence, frameBox });
        }

        std::sort(candidates.begin(), candidates.end(),
            [](const DetrCandidate& left, const DetrCandidate& right) {
                return left.confidence > right.confidence;
            });
        std::vector<DetrCandidate> selected;
        selected.reserve(candidates.size());
        for (const DetrCandidate& candidate : candidates) {
            bool suppressed = false;
            for (const DetrCandidate& kept : selected) {
                if (candidate.classId != kept.classId) continue;
                const int intersection = (candidate.box & kept.box).area();
                const int unionArea = candidate.box.area() + kept.box.area() - intersection;
                if (unionArea > 0 &&
                    static_cast<float>(intersection) / static_cast<float>(unionArea) > _nmsThreshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) selected.push_back(candidate);
        }

        cv::Mat continuousMasks = masks.isContinuous() ? masks : masks.clone();
        for (const DetrCandidate& candidate : selected) {
            const float* maskData = continuousMasks.ptr<float>(0, candidate.index);
            cv::Mat maskLogits(maskHeight, maskWidth, CV_32F, const_cast<float*>(maskData));
            cv::Mat canvasMask;
            cv::resize(maskLogits, canvasMask, cv::Size(_inputWidth, _inputHeight),
                0.0, 0.0, cv::INTER_LINEAR);
            const int contentX = std::clamp(cvRound(padX), 0, _inputWidth - 1);
            const int contentY = std::clamp(cvRound(padY), 0, _inputHeight - 1);
            const int contentWidth = std::clamp(
                std::max(1, cvRound(input.cols * scale)), 1, _inputWidth - contentX);
            const int contentHeight = std::clamp(
                std::max(1, cvRound(input.rows * scale)), 1, _inputHeight - contentY);
            cv::Mat restoredMask;
            cv::resize(canvasMask(cv::Rect(contentX, contentY, contentWidth, contentHeight)),
                restoredMask, input.size(), 0.0, 0.0, cv::INTER_LINEAR);
            cv::Mat binaryMask;
            cv::compare(restoredMask, 0.0, binaryMask, cv::CMP_GT);
            const cv::Mat objectMask = binaryMask(candidate.box).clone();

            ANSCENTER::Object detection;
            detection.classId = candidate.classId;
            detection.className = _classNames[static_cast<size_t>(candidate.classId)];
            detection.confidence = candidate.confidence;
            detection.box = candidate.box;
            detection.mask = std::move(objectMask);
            detection.cameraId = cameraId;
            detections.push_back(std::move(detection));
        }
        return 1;
    }
    catch (const std::exception& error) {
        std::cerr << "[DetrUniformDetector] Inference failed: "
            << error.what() << std::endl;
    }
    detections.clear();
    return 0;
}

void DetrUniformDetector::Destroy() {
    if (_api) {
        if (_session) _api->ReleaseSession(_session);
        if (_memoryInfo) _api->ReleaseMemoryInfo(_memoryInfo);
        if (_sessionOptions) _api->ReleaseSessionOptions(_sessionOptions);
        if (_environment) _api->ReleaseEnv(_environment);
    }
    _session = nullptr;
    _memoryInfo = nullptr;
    _sessionOptions = nullptr;
    _environment = nullptr;
    _api = nullptr;
    if (_runtimeLibrary) {
        FreeLibrary(static_cast<HMODULE>(_runtimeLibrary));
        _runtimeLibrary = nullptr;
    }
    _classNames.clear();
    _outputNames.clear();
    _inputName.clear();
    _usingCuda = false;
}

bool DetrUniformDetector::IsInitialized() const {
    return _session && _api && _memoryInfo && !_classNames.empty() && !_outputNames.empty();
}
