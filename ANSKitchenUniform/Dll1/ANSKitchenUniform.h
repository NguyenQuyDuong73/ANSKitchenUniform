#ifndef ANSKITCHENUNIFORM_H
#define ANSKITCHENUNIFORM_H
#pragma once

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "ANSLIB.h"
#include "ANSByteTrack.h"

#define CUSTOM_API __declspec(dllexport)

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
    };

    std::recursive_mutex _mutex;
    std::unique_ptr<ANSCENTER::ANSLIB> _uniformDetector;
    std::unique_ptr<ANSCENTER::ANSLIB> _poseDetector;
    ANSByteTracker _tracker;

    std::string _uniformLabelMap;
    std::string _uniformClassName;
    std::string _poseLabelMap;
    std::string _poseClassName;
    int _uniformModelType = 23; // ONNXSEG
    int _uniformDetectionType = 2; // SEGMENTATION
    int _poseModelType = 21; // ONNXPOSE
    int _poseDetectionType = 7; // KEYPOINT
    std::string _uniformModelName;
    std::string _poseModelName;
    float _detectionScoreThreshold{ 0.5f };

    std::array<bool, 4> _requiredItems{ true, true, true, true };
    int _alertAfterSecond = 5;
    int _compliantRecheck = 60;
    int _keepTrackObj = 5;
    bool _readParam = false;
    std::map<int, UniformTrackState> _uniformTracks;

public:
    bool Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) override;
    bool OptimizeModel(bool fp16) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input) override;
    std::vector<CustomObject> RunInference(const cv::Mat& input, const std::string& camera_id) override;
    bool ConfigureParameters(CustomParams& param) override;
    bool Destroy() override;
    ANSCustomTL();
    ~ANSCustomTL();
};

#endif
