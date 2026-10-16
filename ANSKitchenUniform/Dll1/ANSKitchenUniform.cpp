#include "ANSKitchenUniform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
//#define FNS_DEBUG

namespace {
    constexpr float kKeypointConfidence = 0.40f;
    constexpr float kWristConfidence = 0.45f;
    constexpr float kFrontThreshold = 0.60f;
    constexpr int kNonCompliantClassId = 4;
    constexpr std::array<const char*, 4> kItemNames{
        "FaceMask", "ChefHat", "KitchenGloves", "Apron"
    };

    struct Keypoint {
        cv::Point2f point;
        float confidence = 0.0f;
        bool visible = false;
    };

    struct PersonRegions {
        bool head = false;
        bool face = false;
        bool torso = false;
        bool leftHand = false;
        bool rightHand = false;
        std::string view = "unknown";
    };

    PersonRegions assessPerson(const ANSCENTER::Object& person, int width, int height) {
        PersonRegions regions;
        const auto getKeypoint = [&](int index, float threshold = kKeypointConfidence) {
            const size_t offset = static_cast<size_t>(index) * 3;
            if (person.kps.size() < offset + 3) return Keypoint{};
            Keypoint result;
            result.point = cv::Point2f(person.kps[offset], person.kps[offset + 1]);
            result.confidence = person.kps[offset + 2];
            result.visible = result.confidence >= threshold &&
                std::isfinite(result.point.x) && std::isfinite(result.point.y);
            return result;
        };
        const auto insideRatio = [&](const cv::Rect2f& raw) {
            const float area = static_cast<float>(raw.area());
            if (area <= 0.0f) return 0.0f;
            const cv::Rect2f frame(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));
            return static_cast<float>((raw & frame).area()) / area;
        };
        const auto pointDistance = [](const Keypoint& first, const Keypoint& second) {
            return static_cast<float>(cv::norm(first.point - second.point));
        };

        const Keypoint nose = getKeypoint(0);
        const Keypoint leftEye = getKeypoint(1);
        const Keypoint rightEye = getKeypoint(2);
        const Keypoint leftEar = getKeypoint(3);
        const Keypoint rightEar = getKeypoint(4);
        const Keypoint leftShoulder = getKeypoint(5);
        const Keypoint rightShoulder = getKeypoint(6);
        const Keypoint leftElbow = getKeypoint(7);
        const Keypoint rightElbow = getKeypoint(8);
        const Keypoint leftWrist = getKeypoint(9, kWristConfidence);
        const Keypoint rightWrist = getKeypoint(10, kWristConfidence);
        const Keypoint leftHip = getKeypoint(11);
        const Keypoint rightHip = getKeypoint(12);

        const std::array<Keypoint, 5> headPoints{ nose, leftEye, rightEye, leftEar, rightEar };
        int headPointCount = 0;
        cv::Point2f headCenter(0.0f, 0.0f);
        for (const auto& point : headPoints) {
            if (!point.visible) continue;
            headCenter += point.point;
            ++headPointCount;
        }

        float headSize = std::max(8.0f, person.box.height * 0.09f);
        if (leftEar.visible && rightEar.visible) headSize = std::max(headSize, pointDistance(leftEar, rightEar) * 1.1f);
        if (leftEye.visible && rightEye.visible) headSize = std::max(headSize, pointDistance(leftEye, rightEye) * 2.6f);
        if (leftShoulder.visible && rightShoulder.visible) headSize = std::max(headSize, pointDistance(leftShoulder, rightShoulder) * 0.55f);
        headSize = std::min(headSize, static_cast<float>(std::max(person.box.width, 8)));

        if (headPointCount > 0) {
            headCenter *= 1.0f / static_cast<float>(headPointCount);
            const float halfWidth = headSize * 0.8f;
            const cv::Rect2f headRaw(
                headCenter.x - halfWidth, headCenter.y - 1.5f * headSize,
                2.0f * halfWidth, 2.4f * headSize);
            const bool hasTwoHeadPoints = headPointCount >= 2;
            const bool shoulderFallback = leftShoulder.visible && rightShoulder.visible && headPointCount >= 1;
            regions.head = (hasTwoHeadPoints || shoulderFallback) &&
                insideRatio(headRaw) >= 0.60f;
            const cv::Rect2f crown(
                headCenter.x - 0.35f * headSize, headCenter.y - 0.8f * headSize,
                0.70f * headSize, 0.60f * headSize);
            regions.head = regions.head && insideRatio(crown) >= 0.60f;
        }

        regions.face = regions.head && (nose.visible || leftEye.visible || rightEye.visible);   

        const int shoulderCount = static_cast<int>(leftShoulder.visible) + static_cast<int>(rightShoulder.visible);
        const int hipCount = static_cast<int>(leftHip.visible) + static_cast<int>(rightHip.visible);
        if (shoulderCount > 0 && shoulderCount + hipCount >= 2) {
            const float shoulderY = std::min(
                leftShoulder.visible ? leftShoulder.point.y : std::numeric_limits<float>::max(),
                rightShoulder.visible ? rightShoulder.point.y : std::numeric_limits<float>::max());
            float shoulderWidth = leftShoulder.visible && rightShoulder.visible
                ? pointDistance(leftShoulder, rightShoulder)
                : std::max(0.3f * person.box.width, 0.25f * person.box.height);
            float hipY = shoulderY + 1.4f * shoulderWidth;
            if (hipCount > 0) {
                float sum = 0.0f;
                if (leftHip.visible) sum += leftHip.point.y;
                if (rightHip.visible) sum += rightHip.point.y;
                hipY = sum / static_cast<float>(hipCount);
                if (hipY <= shoulderY) hipY = shoulderY + 0.8f * shoulderWidth;
            }
            const float torsoWidth = std::max(shoulderWidth, 8.0f);
            const cv::Rect2f torso(
                person.box.x + 0.15f * person.box.width,
                shoulderY - 0.1f * (hipY - shoulderY),
                0.70f * person.box.width,
                (hipY - shoulderY) * 1.55f);
            regions.torso = torsoWidth > 0.0f && insideRatio(torso) >= 0.60f;
        }

        auto handVisible = [&](const Keypoint& elbow, const Keypoint& wrist) {
            if (!elbow.visible || !wrist.visible) return false;
            const float forearm = std::max(pointDistance(elbow, wrist), 0.9f * headSize);
            const cv::Point2f direction = (wrist.point - elbow.point) /
                std::max(pointDistance(elbow, wrist), 1e-6f);
            const cv::Point2f center = wrist.point + 0.3f * forearm * direction;
            const float halfWidth = 0.65f * forearm;
            const cv::Rect2f hand(center.x - halfWidth, center.y - halfWidth, 2.0f * halfWidth, 2.0f * halfWidth);
            return insideRatio(hand) >= 0.60f;
        };
        regions.leftHand = handVisible(leftElbow, leftWrist);
        regions.rightHand = handVisible(rightElbow, rightWrist);

        std::optional<float> bodyScore;
        if (leftShoulder.visible && rightShoulder.visible) {
            cv::Point2f spine(0.0f, 1.0f);
            float torsoLength = 0.0f;
            if (hipCount > 0) {
                cv::Point2f hipCenter(0.0f, 0.0f);
                if (leftHip.visible) hipCenter += leftHip.point;
                if (rightHip.visible) hipCenter += rightHip.point;
                hipCenter *= 1.0f / static_cast<float>(hipCount);
                const cv::Point2f shoulderCenter = (leftShoulder.point + rightShoulder.point) * 0.5f;
                spine = hipCenter - shoulderCenter;
                torsoLength = static_cast<float>(cv::norm(spine));
            } else if (headPointCount > 0) {
                torsoLength = 3.0f * headSize;
            }
            if (torsoLength > 1e-6f) {
                spine *= 1.0f / std::max(static_cast<float>(cv::norm(spine)), 1e-6f);
                const cv::Point2f normal(-spine.y, spine.x);
                const float shoulderWidth = std::abs((leftShoulder.point - rightShoulder.point).dot(normal));
                const float ratio = shoulderWidth / torsoLength;
                bodyScore = std::clamp((ratio - 0.25f) / (0.80f - 0.25f), 0.0f, 1.0f);
            }
        }

        float faceScore = -1.0f;
        if (nose.visible) {
            faceScore = 0.20f;
            if (leftEye.visible && rightEye.visible) {
                const float dx = leftEye.point.x - rightEye.point.x;
                if (std::abs(dx) >= 1e-6f) {
                    const float t = (nose.point.x - rightEye.point.x) / dx;
                    faceScore = std::max(0.20f,
                        std::clamp(1.0f - 2.0f * std::abs(t - 0.5f), 0.0f, 1.0f));
                }
            }
        }
        if (leftShoulder.visible && rightShoulder.visible &&
            std::abs(leftShoulder.point.x - rightShoulder.point.x) > 0.15f * person.box.width &&
            leftShoulder.point.x - rightShoulder.point.x < 0.0f) {
            regions.view = "back";
        } else if (bodyScore.has_value()) {
            regions.view = *bodyScore >= kFrontThreshold ? "front" : "side";
        } else if (faceScore >= 0.0f) {
            regions.view = faceScore >= kFrontThreshold ? "front" : "side";
        }
        return regions;
    }

    int classifyAllowedColor(const cv::Mat& frame, const ANSCENTER::Object& detection) {
        if (frame.empty() || detection.box.area() <= 0) return -1;
        const cv::Rect box = detection.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.width <= 0 || box.height <= 0) return -1;
        const cv::Mat pixels = frame(box);
        cv::Mat itemMask;

        if (!detection.mask.empty()) {
            cv::Mat mask = detection.mask;
            if (mask.channels() > 1) cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
            if (mask.size() == frame.size()) {
                itemMask = mask(box).clone();
            } else {
                cv::resize(mask, itemMask, box.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            cv::threshold(itemMask, itemMask, 0.5, 255.0, cv::THRESH_BINARY);
            itemMask.convertTo(itemMask, CV_8U);
            if (cv::countNonZero(itemMask) > 0) {
                cv::Mat eroded;
                cv::erode(itemMask, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
                if (cv::countNonZero(eroded) > 0) itemMask = std::move(eroded);
            }
        } else {
            itemMask = cv::Mat(box.size(), CV_8U, cv::Scalar(0));
            const cv::Rect core(
                std::max(0, static_cast<int>(box.width * 0.20f)),
                std::max(0, static_cast<int>(box.height * 0.20f)),
                std::max(1, static_cast<int>(box.width * 0.60f)),
                std::max(1, static_cast<int>(box.height * 0.60f)));
            cv::rectangle(itemMask, core & cv::Rect(0, 0, box.width, box.height), cv::Scalar(255), cv::FILLED);
        }

        if (cv::countNonZero(itemMask) < 50) return -1;
        cv::Mat hsv;
        cv::cvtColor(pixels, hsv, cv::COLOR_BGR2HSV);
        cv::Mat lab;
        cv::cvtColor(pixels, lab, cv::COLOR_BGR2Lab);

        int allowedPixels = 0;
        int redPixels = 0;
        for (int y = 0; y < itemMask.rows; ++y) {
            const auto* maskRow = itemMask.ptr<unsigned char>(y);
            const auto* hsvRow = hsv.ptr<cv::Vec3b>(y);
            const auto* labRow = lab.ptr<cv::Vec3b>(y);
            for (int x = 0; x < itemMask.cols; ++x) {
                if (maskRow[x] == 0) continue;
                const int h = hsvRow[x][0];
                const int s = hsvRow[x][1];
                const int v = hsvRow[x][2];
                const int l = labRow[x][0];
                const int a = labRow[x][1];
                const int b = labRow[x][2];
                const bool red = ((h <= 5 || h >= 170) && s >= 80 && v >= 60);
                const bool white = (s <= 60 && v >= 120) ||
                    (l >= 170 && std::abs(a - 128) <= 14 && std::abs(b - 128) <= 14);
                if (red || white) ++allowedPixels;
                if (red) ++redPixels;
            }
        }
        const int totalPixels = cv::countNonZero(itemMask);
        if (totalPixels < 50) return -1;
        return static_cast<float>(allowedPixels) / totalPixels >= 0.70f ? 1 : 0;
    }

}

ANSCustomTL::ANSCustomTL() = default;

bool ANSCustomTL::OptimizeModel(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    if (!_uniformDetector || !_poseDetector) {
        std::cerr << "[ANSKitchenUniform] Cannot optimize before model initialization." << std::endl;
        return false;
    }
    const bool uniformOptimized = _uniformDetector->Optimize(fp16);
    const bool poseOptimized = _poseDetector->Optimize(fp16);
    if (!uniformOptimized || !poseOptimized) {
        std::cerr << "[ANSKitchenUniform] Model optimization failed." << std::endl;
        return false;
    }
    return true;
}

std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat& input) {
    return RunInference(input, "CustomCam");
}

bool ANSCustomTL::Destroy() {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _uniformDetector.reset();
    _poseDetector.reset();
    _uniformTracks.clear();
    _readParam = false;
    return true;
}

bool ANSCustomTL::Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    try {
        _modelDirectory = modelDirectory;
        _detectionScoreThreshold = detectionScoreThreshold;
        _uniformModelName = "uniform";
        _uniformClassName = "uniform.names";
        _poseModelName = "pose";
        _poseClassName = "pose.names";
        _uniformModelType = 23; // ONNXSEG
        _uniformDetectionType = 2; // SEGMENTATION
        _poseModelType = 21; // ONNXPOSE
        _poseDetectionType = 7; // KEYPOINT
        _uniformDetector = std::make_unique<ANSCENTER::ANSLIB>();
        _poseDetector = std::make_unique<ANSCENTER::ANSLIB>();

        constexpr float uniformModelConfidence = 0.25f;
        constexpr float uniformNmsThreshold = 0.50f;
        constexpr float poseModelConfidence = 0.25f;
        constexpr float poseNmsThreshold = 0.50f;
        constexpr int autoEngineDetection = 1;
//#ifdef FNS_DEBUG
        _loadEngineOnCreate = true;
//#endif
        const int loadEngineOnCreation = _loadEngineOnCreate ? 1 : 0;
        const std::string licenseKey;

        const int uniformResult = _uniformDetector->LoadModelFromFolder(
            licenseKey.c_str(), _uniformModelName.c_str(), _uniformClassName.c_str(),
            _detectionScoreThreshold, uniformModelConfidence, uniformNmsThreshold,
            autoEngineDetection, _uniformModelType, _uniformDetectionType, loadEngineOnCreation,
            _modelDirectory.c_str(), _uniformLabelMap);
        const int poseResult = _poseDetector->LoadModelFromFolder(
            licenseKey.c_str(), _poseModelName.c_str(), _poseClassName.c_str(),
            _detectionScoreThreshold, poseModelConfidence, poseNmsThreshold,
            autoEngineDetection, _poseModelType, _poseDetectionType, loadEngineOnCreation,
            _modelDirectory.c_str(), _poseLabelMap);
        if (uniformResult != 1 || poseResult != 1) {
            std::cerr << "[ANSKitchenUniform] Model loading failed: uniform=" << uniformResult
                << ", pose=" << poseResult << std::endl;
            Destroy();
            return false;
        }

        const int uniformTrackerResult = _uniformDetector->SetTracker(BYTETRACK, 1);
        const int poseTrackerResult = _poseDetector->SetTracker(BYTETRACK, 1);
        if (uniformTrackerResult != 1 || poseTrackerResult != 1) {
            std::cerr << "[ANSKitchenUniform] Tracker initialization warning: uniform="
                << uniformTrackerResult << ", pose=" << poseTrackerResult << std::endl;
        }

        labelMap = "FaceMask,ChefHat,KitchenGloves,Apron,Non_Compliant";
        _uniformTracks.clear();
        return true;
    } catch (const std::exception& error) {
        std::cerr << "[ANSKitchenUniform] Initialization error: " << error.what() << std::endl;
        Destroy();
        return false;
    }
}

ANSCustomTL::~ANSCustomTL() {
    Destroy();
}

bool ANSCustomTL::ConfigureParameters(CustomParams& param) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    param.Parameters.clear();
    constexpr std::array<const char*, 4> parameterItemNames{
        "ChefHat", "FaceMask", "KitchenGloves", "Apron"
    };
    for (const char* itemName : parameterItemNames) {
        CustomParameter item;
        item.Name = itemName;
        item.DataType = "Boolean";
        item.NoOfDecimals = 0;
        item.MaxValue = 0;
        item.MinValue = 0;
        item.StartValue = "true";
        item.DefaultValue = "true";
        item.Value.clear();
        param.Parameters.push_back(std::move(item));
    }

    CustomParameter alertAfterSecond;
    alertAfterSecond.Name = "AlertAfterSecond";
    alertAfterSecond.DataType = "Integer";
    alertAfterSecond.NoOfDecimals = 0;
    alertAfterSecond.MaxValue = 1000000;
    alertAfterSecond.MinValue = 0;
    alertAfterSecond.StartValue = "5";
    alertAfterSecond.DefaultValue = "5";
    alertAfterSecond.Value.clear();
    param.Parameters.push_back(std::move(alertAfterSecond));

    CustomParameter compliantRecheck;
    compliantRecheck.Name = "CompliantRecheck";
    compliantRecheck.DataType = "Integer";
    compliantRecheck.NoOfDecimals = 0;
    compliantRecheck.MaxValue = 1000000;
    compliantRecheck.MinValue = 0;
    compliantRecheck.StartValue = "60";
    compliantRecheck.DefaultValue = "60";
    compliantRecheck.Value.clear();
    param.Parameters.push_back(std::move(compliantRecheck));

    CustomParameter keepTrackObj;
    keepTrackObj.Name = "KeepTrackObj";
    keepTrackObj.DataType = "Integer";
    keepTrackObj.NoOfDecimals = 0;
    keepTrackObj.MaxValue = 1000000;
    keepTrackObj.MinValue = 0;
    keepTrackObj.StartValue = "5";
    keepTrackObj.DefaultValue = "5";
    keepTrackObj.Value.clear();
    param.Parameters.push_back(std::move(keepTrackObj));

    _readParam = false;
    return true;
}

std::vector<CustomObject> ANSCustomTL::RunInference(const cv::Mat& input, const std::string& camera_id) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    std::vector<CustomObject> results;
    if (input.empty()) {
        std::cerr << "[ANSKitchenUniform] Skipping an empty input frame." << std::endl;
        return results;
    }
    if (!_uniformDetector || !_poseDetector) {
        std::cerr << "[ANSKitchenUniform] RunInference called before model initialization." << std::endl;
        return results;
    }

    try {
        if (!_readParam) {
            const auto parseBoolean = [](std::string value) {
                std::transform(value.begin(), value.end(), value.begin(),
                    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                return value == "TRUE" || value == "YES" || value == "1";
            };
            for (size_t index = 0; index < kItemNames.size(); ++index) {
                const std::string requestedName = kItemNames[index];
                auto param = std::find_if(_params.Parameters.begin(), _params.Parameters.end(),
                    [&](const CustomParameter& item) { return item.Name == requestedName; });
                if (param != _params.Parameters.end()) {
                    const std::string value = !param->Value.empty() ? param->Value : param->DefaultValue;
                    _requiredItems[index] = value.empty() || parseBoolean(value);
                }
            }
            auto readInteger = [&](const std::string& name, int fallback, int maximum) {
                auto param = std::find_if(_params.Parameters.begin(), _params.Parameters.end(),
                    [&](const CustomParameter& item) { return item.Name == name; });
                if (param == _params.Parameters.end()) return fallback;
                const std::string value = !param->Value.empty() ? param->Value : param->DefaultValue;
                if (value.empty()) return fallback;
                try {
                    return std::clamp(std::stoi(value), 0, maximum);
                } catch (const std::exception&) {
                    std::cerr << "[ANSKitchenUniform] Invalid integer parameter " << name
                        << "=" << value << "; using " << fallback << "." << std::endl;
                    return fallback;
                }
            };
            _alertAfterSecond = readInteger("AlertAfterSecond", 5, 1000000);
            _compliantRecheck = readInteger("CompliantRecheck", 60, 1000000);
            _keepTrackObj = readInteger("KeepTrackObj", 5, 1000000);
            _readParam = true;
        }

        const auto currentTime = std::chrono::steady_clock::now();
        const auto keepTrackDuration = std::chrono::seconds(_keepTrackObj);
        std::vector<ANSCENTER::Object> people;
        const int poseStatus = _poseDetector->RunInference(input, camera_id.c_str(), people);
        if (poseStatus != 1) {
            std::cerr << "[ANSKitchenUniform] Pose inference failed with status " << poseStatus << "." << std::endl;
            return results;
        }
        if (people.empty()) {
            for (auto it = _uniformTracks.begin(); it != _uniformTracks.end();) {
                it->second.wasFrontMissing = false;
                if (it->second.hasSeen && currentTime - it->second.lastSeenTime > keepTrackDuration) {
                    it = _uniformTracks.erase(it);
                }
                else ++it;
            }
            return results;
        }

        std::vector<UniformTrackState*> trackStates;
        trackStates.reserve(people.size());
        bool shouldRunUniform = false;
        std::set<int> currentTrackIds;
        for (const auto& person : people) {
            currentTrackIds.insert(person.trackId);
            UniformTrackState& state = _uniformTracks[person.trackId];
            if (state.hasSeen && currentTime - state.lastSeenTime > keepTrackDuration) {
                state = UniformTrackState{};
            }
            state.lastSeenTime = currentTime;
            state.hasSeen = true;
            trackStates.push_back(&state);
            const bool recheckDue = _compliantRecheck == 0 || !state.hasUniformCheck ||
                currentTime - state.lastUniformCheckTime >= std::chrono::seconds(_compliantRecheck);
            if (!state.compliant || recheckDue) shouldRunUniform = true;
        }

        for (auto it = _uniformTracks.begin(); it != _uniformTracks.end();) {
            if (currentTrackIds.find(it->first) == currentTrackIds.end()) {
                it->second.wasFrontMissing = false;
            }
            if (currentTime - it->second.lastSeenTime > keepTrackDuration) {
                it = _uniformTracks.erase(it);
            } else {
                ++it;
            }
        }

        if (!shouldRunUniform) return results;

        std::vector<ANSCENTER::Object> uniformItems;
        const int uniformStatus = _uniformDetector->RunInference(input, camera_id.c_str(), uniformItems);
        if (uniformStatus != 1) {
            std::cerr << "[ANSKitchenUniform] Uniform inference failed with status " << uniformStatus << "." << std::endl;
            return results;
        }

        const auto bestDetection = [&](int wantedItem, const cv::Rect& personBox) -> const ANSCENTER::Object* {
            const ANSCENTER::Object* best = nullptr;
            for (const auto& detection : uniformItems) {
                std::string name = detection.className;
                name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) {
                    return c == '_' || c == ' ' || c == '-';
                }), name.end());
                std::transform(name.begin(), name.end(), name.begin(),
                    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

                int detectedItem = -1;
                for (size_t item = 0; item < kItemNames.size(); ++item) {
                    std::string expected = kItemNames[item];
                    std::transform(expected.begin(), expected.end(), expected.begin(),
                        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                    if (name == expected) {
                        detectedItem = static_cast<int>(item);
                        break;
                    }
                }
                if (detection.className.empty() && detection.classId >= 0 && detection.classId < 4) {
                    detectedItem = detection.classId;
                }
                if (detectedItem != wantedItem || detection.box.area() <= 0 || personBox.area() <= 0) continue;
                const cv::Point center(
                    detection.box.x + detection.box.width / 2,
                    detection.box.y + detection.box.height / 2);
                const bool belongsToPerson = personBox.contains(center) ||
                    (detection.box & personBox).area() >= 0.25 * detection.box.area();
                if (!belongsToPerson) continue;
                if (best == nullptr || detection.confidence > best->confidence) best = &detection;
            }
            return best;
        };

        for (size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
            const auto& person = people[personIndex];
            UniformTrackState& state = *trackStates[personIndex];
            const bool recheckDue = _compliantRecheck == 0 || !state.hasUniformCheck ||
                currentTime - state.lastUniformCheckTime >= std::chrono::seconds(_compliantRecheck);
            if (state.compliant && !recheckDue) continue;
            if (recheckDue && state.compliant) {
                state.confirmedItems.clear();
                state.compliant = false;
                state.frontMissingDuration = std::chrono::duration<double>::zero();
                state.wasFrontMissing = false;
                state.alertRaised = false;
            }
            state.lastUniformCheckTime = currentTime;
            state.hasUniformCheck = true;

            const PersonRegions regions = assessPerson(person, input.cols, input.rows);
            bool visibleViolation = false;
            for (size_t item = 0; item < kItemNames.size(); ++item) {
                if (!_requiredItems[item]) continue;
                const ANSCENTER::Object* detection = bestDetection(static_cast<int>(item), person.box);
                const bool found = detection != nullptr;
                int colorStatus = 1;
                if (found && (item == 1 || item == 3)) {
                    colorStatus = classifyAllowedColor(input, *detection);
                }
                const bool itemVisible = item == 0 ? regions.face
                    : item == 1 ? regions.head
                    : item == 2 ? (regions.leftHand || regions.rightHand)
                    : regions.torso;

                if (found && colorStatus == 1) {
                    state.confirmedItems.insert(static_cast<int>(item));
                } else if (found && colorStatus == 0) {
                    state.confirmedItems.erase(static_cast<int>(item));
                    visibleViolation = true;
                } else if (itemVisible && !found) {
                    state.confirmedItems.erase(static_cast<int>(item));
                    visibleViolation = true;
                } else if (itemVisible && found && colorStatus < 0) {
                    state.confirmedItems.erase(static_cast<int>(item));
                }
            }

            const int requiredCount = static_cast<int>(std::count(_requiredItems.begin(), _requiredItems.end(), true));
            state.compliant = requiredCount > 0 &&
                static_cast<int>(state.confirmedItems.size()) >= requiredCount;
            if (state.compliant) {
                state.frontMissingDuration = std::chrono::duration<double>::zero();
                state.wasFrontMissing = false;
                state.alertRaised = false;
                continue;
            }

            const bool missingFrontNow = regions.view == "front" && visibleViolation;
            if (missingFrontNow) {
                if (state.wasFrontMissing) {
                    state.frontMissingDuration += currentTime - state.lastMissingFrontTime;
                }
                state.lastMissingFrontTime = currentTime;
            }
            state.wasFrontMissing = missingFrontNow;
            if (state.frontMissingDuration > std::chrono::seconds(_alertAfterSecond) && !state.alertRaised) {
                CustomObject event;
                event.classId = kNonCompliantClassId;
                event.className = "Non_Compliant";
                event.confidence = person.confidence;
                event.box = person.box & cv::Rect(0, 0, input.cols, input.rows);
                event.kps = person.kps;
                event.cameraId = camera_id;
                event.trackId = person.trackId;
                results.push_back(std::move(event));
                state.alertRaised = true;
            }
        }

    } catch (const std::exception& error) {
        std::cerr << "[ANSKitchenUniform] Inference error: " << error.what() << std::endl;
    }
    return results;
}
