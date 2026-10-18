#ifndef ANSKITCHENUNIFORM_H
#define ANSKITCHENUNIFORM_H
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "ANSLIB.h"
#include "ANSByteTrack.h"
#include "third_party/onnxruntime/onnxruntime_c_api.h"

#define CUSTOM_API __declspec(dllexport)

class DetrUniformDetector
{
private:
    void* _runtimeLibrary{ nullptr };
    const OrtApi* _api{ nullptr };
    OrtEnv* _environment{ nullptr };
    OrtSessionOptions* _sessionOptions{ nullptr };
    OrtSession* _session{ nullptr };
    OrtMemoryInfo* _memoryInfo{ nullptr };
    std::vector<std::string> _classNames;
    std::vector<std::string> _outputNames;
    std::string _inputName;
    float _scoreThreshold{ 0.25f };
    float _nmsThreshold{ 0.50f };
    int _inputWidth{ 432 };
    int _inputHeight{ 432 };
    bool _usingCuda{ false };

public:
    bool Initialize(const std::string& modelPath, const std::string& classNamesPath, float scoreThreshold);
    int RunInference(const cv::Mat& input, const std::string& cameraId, std::vector<ANSCENTER::Object>& detections);
    void Destroy();
    bool IsInitialized() const;
};

struct UniformDebugPersonFrame {
    int trackId = -1;
    cv::Rect box;
    std::string verdict;
    std::array<std::string, 4> itemStatuses{};
    std::vector<std::string> detections;
    cv::Mat annotatedCrop;
};

class CUSTOM_API ANSCustomTL : public IANSCustomClass
{
private:
    struct UniformTrackState {
        std::chrono::steady_clock::time_point lastSeenTime{};
        std::chrono::steady_clock::time_point lastUniformCheckTime{};
        std::chrono::steady_clock::time_point lastMissingFrontTime{};
        std::chrono::duration<double> frontMissingDuration{};
        bool hasSeen = false;
        bool hasUniformCheck = false;
        bool wasFrontMissing = false;
        bool compliant = false;
        bool alertRaised = false;
        std::set<int> confirmedItems;
        std::array<std::deque<int>, 4> itemStatusHistory;
        std::array<std::int64_t, 4> lastPositiveFrame{ -1, -1, -1, -1 };
        std::array<std::string, 4> debugItemStatuses{};
        std::array<std::string, 4> debugItemColors{};
    };

    std::recursive_mutex _mutex;
    std::unique_ptr<DetrUniformDetector> _uniformDetector;
    std::unique_ptr<ANSCENTER::ANSLIB> _poseDetector;
    ANSByteTracker _tracker;

    std::string _uniformLabelMap;
    std::string _poseLabelMap;
    std::string _poseClassName;
    int _poseModelType; // ONNXPOSE
    int _poseDetectionType; // KEYPOINT
    std::string _uniformModelName;
    std::string _poseModelName;
    float _detectionScoreThreshold{ 0.5f };

    std::array<bool, 4> _requiredItems{ true, true, true, true };
    int _alertAfterSecond = 5;
    int _compliantRecheck = 60;
    int _keepTrackObj = 5;
    std::int64_t _inferenceFrameIndex = 0;
    bool _readParam = false;
    bool _debugVisualizationEnabled = false;
    cv::Mat _debugFrame;
    std::vector<UniformDebugPersonFrame> _debugPersonFrames;
    double _debugPoseInferenceMilliseconds = 0.0;
    double _debugUniformInferenceMilliseconds = 0.0;
    bool _debugUniformInferenceRan = false;
    std::map<int, UniformTrackState> _uniformTracks;

public:
    bool Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) override;
    bool OptimizeModel(bool fp16) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input, const std::string& camera_id) override;
    bool ConfigureParameters(CustomParams& param) override;
    bool Destroy() override;
    void SetDebugVisualization(bool enabled);
    bool GetDebugFrame(cv::Mat& frame);
    bool GetDebugPersonFrames(std::vector<UniformDebugPersonFrame>& frames);
    void GetDebugTimings(double& poseMilliseconds, double& uniformMilliseconds, bool& uniformRan);
    ANSCustomTL();
    ~ANSCustomTL();
};

#endif
