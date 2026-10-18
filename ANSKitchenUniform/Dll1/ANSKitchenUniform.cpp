#include "ANSKitchenUniform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
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
    constexpr float kWristConfidence = 0.60f;
    constexpr float kFrontThreshold = 0.60f;
    constexpr float kMinPersonHeightRatio = 0.15f;
    constexpr int kMinPersonBoxArea = 2500;
    constexpr int kNonCompliantClassId = 4;
    constexpr std::array<const char*, 4> kItemNames{
        "FaceMask", "ChefHat", "KitchenGloves", "Apron"
    };
    const std::array<cv::Scalar, 4> kItemColors{
        cv::Scalar(255, 200, 0), cv::Scalar(0, 215, 255),
        cv::Scalar(255, 0, 200), cv::Scalar(0, 140, 255)
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
        cv::Rect2f headBox;
        cv::Rect2f crownBox;
        cv::Rect2f torsoBox;
        cv::Rect2f leftHandBox;
        cv::Rect2f rightHandBox;
        std::optional<float> bodyFacing;
        std::optional<float> faceFacing;
    };

    bool pointInExpandedRegion(const cv::Point& point, const cv::Rect2f& region,
        float paddingRatio = 0.20f) {
        if (region.width <= 0.0f || region.height <= 0.0f) return false;
        const float padX = region.width * paddingRatio;
        const float padY = region.height * paddingRatio;
        return point.x >= region.x - padX && point.x <= region.x + region.width + padX &&
            point.y >= region.y - padY && point.y <= region.y + region.height + padY;
    }

    bool detectionMatchesPoseRegion(int itemIndex, const cv::Point& center,
        const PersonRegions& regions) {
        switch (itemIndex) {
        case 0:
            return regions.face && pointInExpandedRegion(center, regions.headBox);
        case 1:
            return regions.head && pointInExpandedRegion(center, regions.crownBox, 0.35f);
        case 2:
            return (regions.leftHand && pointInExpandedRegion(center, regions.leftHandBox)) ||
                (regions.rightHand && pointInExpandedRegion(center, regions.rightHandBox));
        case 3:
            return regions.torso && pointInExpandedRegion(center, regions.torsoBox);
        default:
            return false;
        }
    }

    int detectionItemIndex(const ANSCENTER::Object& detection) {
        if (detection.className.empty() && detection.classId >= 0 && detection.classId < 4) {
            return detection.classId;
        }
        std::string name = detection.className;
        name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char character) {
            return character == '_' || character == ' ' || character == '-';
            }), name.end());
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
            });
        for (size_t index = 0; index < kItemNames.size(); ++index) {
            std::string expected = kItemNames[index];
            std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char character) {
                return static_cast<char>(std::toupper(character));
                });
            if (name == expected) return static_cast<int>(index);
        }
        return -1;
    }

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
            const cv::Rect2f clipped = raw & frame;
            if (std::min(clipped.width, clipped.height) < 12.0f) return 0.0f;
            return static_cast<float>(clipped.area()) / area;
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

        float headSize = person.box.height * 0.09f;
        bool hasHeadSizeCue = false;
        if (leftEar.visible && rightEar.visible) {
            headSize = std::max(headSize, pointDistance(leftEar, rightEar) * 1.1f);
            hasHeadSizeCue = true;
        }
        if (leftEye.visible && rightEye.visible) {
            headSize = std::max(headSize, pointDistance(leftEye, rightEye) * 2.6f);
            hasHeadSizeCue = true;
        }
        if (leftShoulder.visible && rightShoulder.visible) {
            headSize = std::max(headSize, pointDistance(leftShoulder, rightShoulder) * 0.55f);
            hasHeadSizeCue = true;
        }
        if (!hasHeadSizeCue) headSize = person.box.height * 0.14f;
        headSize = std::max(headSize, 8.0f);
        headSize = std::min(headSize, static_cast<float>(std::max(person.box.width, 8)));

        if (headPointCount > 0) {
            headCenter *= 1.0f / static_cast<float>(headPointCount);
            const float halfWidth = headSize * 0.8f;
            const cv::Rect2f headRaw(
                headCenter.x - halfWidth, headCenter.y - 1.5f * headSize,
                2.0f * halfWidth, 2.4f * headSize);
            regions.headBox = headRaw;
            const bool hasTwoHeadPoints = headPointCount >= 2;
            const bool shoulderFallback = leftShoulder.visible && rightShoulder.visible && headPointCount >= 1;
            regions.head = (hasTwoHeadPoints || shoulderFallback) &&
                insideRatio(headRaw) >= 0.60f;
            const cv::Rect2f crown(
                headCenter.x - 0.35f * headSize, headCenter.y - 0.8f * headSize,
                0.70f * headSize, 0.60f * headSize);
            regions.crownBox = crown;
            regions.head = regions.head && insideRatio(crown) >= 0.60f;
        }

        regions.face = regions.head && (nose.visible || leftEye.visible || rightEye.visible);

        const int shoulderCount = static_cast<int>(leftShoulder.visible) + static_cast<int>(rightShoulder.visible);
        const int hipCount = static_cast<int>(leftHip.visible) + static_cast<int>(rightHip.visible);
        if (shoulderCount > 0 && shoulderCount + hipCount >= 2) {
            const float shoulderY = std::min(
                leftShoulder.visible ? leftShoulder.point.y : std::numeric_limits<float>::max(),
                rightShoulder.visible ? rightShoulder.point.y : std::numeric_limits<float>::max());
            const float shoulderWidth = leftShoulder.visible && rightShoulder.visible
                ? std::max(pointDistance(leftShoulder, rightShoulder), 4.0f)
                : std::max(std::min(0.3f * person.box.width, 0.25f * person.box.height), 4.0f);
            float torsoHeight = 0.0f;
            float hipY = 0.0f;
            float expandDown = 0.0f;
            if (hipCount > 0) {
                float sum = 0.0f;
                if (leftHip.visible) sum += leftHip.point.y;
                if (rightHip.visible) sum += rightHip.point.y;
                hipY = sum / static_cast<float>(hipCount);
                torsoHeight = std::max(hipY - shoulderY, 0.8f * shoulderWidth);
                hipY = shoulderY + torsoHeight;
                expandDown = 0.55f;
            }
            else {
                torsoHeight = 1.4f * shoulderWidth;
                hipY = shoulderY + torsoHeight;
            }
            std::vector<float> torsoXs;
            if (leftShoulder.visible) torsoXs.push_back(leftShoulder.point.x);
            if (rightShoulder.visible) torsoXs.push_back(rightShoulder.point.x);
            if (leftHip.visible) torsoXs.push_back(leftHip.point.x);
            if (rightHip.visible) torsoXs.push_back(rightHip.point.x);
            const auto [minX, maxX] = std::minmax_element(torsoXs.begin(), torsoXs.end());
            const float widthFromKeypoints = torsoXs.size() > 1 ? *maxX - *minX : 0.0f;
            const float torsoWidth = std::max(widthFromKeypoints, shoulderWidth);
            const float centerX = torsoXs.size() > 1 ? (*minX + *maxX) * 0.5f : torsoXs.front();
            const float halfWidth = 0.7f * torsoWidth;
            const cv::Rect2f torso(
                centerX - halfWidth,
                shoulderY - 0.1f * torsoHeight,
                2.0f * halfWidth,
                hipY + expandDown * torsoHeight - (shoulderY - 0.1f * torsoHeight));
            regions.torsoBox = torso;
            regions.torso = torsoWidth > 0.0f && insideRatio(torso) >= 0.60f;
        }

        auto handVisible = [&](const Keypoint& elbow, const Keypoint& wrist, cv::Rect2f& handBox) {
            if (!elbow.visible || !wrist.visible) return false;
            const float forearm = std::max(pointDistance(elbow, wrist), 0.9f * headSize);
            const cv::Point2f direction = (wrist.point - elbow.point) /
                std::max(pointDistance(elbow, wrist), 1e-6f);
            const cv::Point2f center = wrist.point + 0.3f * forearm * direction;
            const float halfWidth = 0.65f * forearm;
            handBox = cv::Rect2f(center.x - halfWidth, center.y - halfWidth,
                2.0f * halfWidth, 2.0f * halfWidth);
            return insideRatio(handBox) >= 0.60f;
            };
        regions.leftHand = handVisible(leftElbow, leftWrist, regions.leftHandBox);
        regions.rightHand = handVisible(rightElbow, rightWrist, regions.rightHandBox);

        std::optional<float> bodyScore;
        if (leftShoulder.visible && rightShoulder.visible) {
            cv::Point2f spine(0.0f, 1.0f);
            float torsoLength = 0.0f;
            float noHipPenalty = 1.0f;
            if (hipCount > 0) {
                cv::Point2f hipCenter(0.0f, 0.0f);
                if (leftHip.visible) hipCenter += leftHip.point;
                if (rightHip.visible) hipCenter += rightHip.point;
                hipCenter *= 1.0f / static_cast<float>(hipCount);
                const cv::Point2f shoulderCenter = (leftShoulder.point + rightShoulder.point) * 0.5f;
                spine = hipCenter - shoulderCenter;
                torsoLength = static_cast<float>(cv::norm(spine));
            }
            else {
                float faceHeadSize = 0.0f;
                if (leftEar.visible && rightEar.visible) {
                    faceHeadSize = std::max(faceHeadSize, pointDistance(leftEar, rightEar) * 1.1f);
                }
                if (leftEye.visible && rightEye.visible) {
                    faceHeadSize = std::max(faceHeadSize, pointDistance(leftEye, rightEye) * 2.6f);
                }
                if (faceHeadSize > 0.0f) {
                    torsoLength = 3.0f * faceHeadSize;
                    noHipPenalty = 0.85f;
                }
            }
            if (torsoLength > 1e-6f) {
                spine *= 1.0f / std::max(static_cast<float>(cv::norm(spine)), 1e-6f);
                const cv::Point2f normal(-spine.y, spine.x);
                const float shoulderWidth = std::abs((leftShoulder.point - rightShoulder.point).dot(normal));
                const auto frontalScore = [](float ratio, float sideThreshold, float frontThreshold) {
                    return std::clamp((ratio - sideThreshold) /
                        std::max(frontThreshold - sideThreshold, 1e-6f), 0.0f, 1.0f);
                    };
                float scoreSum = frontalScore(shoulderWidth / torsoLength, 0.25f, 0.80f);
                int scoreCount = 1;
                if (leftHip.visible && rightHip.visible) {
                    const float hipWidth = std::abs((leftHip.point - rightHip.point).dot(normal));
                    scoreSum += frontalScore(hipWidth / torsoLength, 0.15f, 0.55f);
                    ++scoreCount;
                }
                bodyScore = noHipPenalty * scoreSum / static_cast<float>(scoreCount);
            }
        }

        float faceScore = -1.0f;
        if (nose.visible) {
            std::vector<float> cues;
            const std::array<std::pair<Keypoint, Keypoint>, 2> pairs{
                std::pair<Keypoint, Keypoint>{ leftEye, rightEye },
                std::pair<Keypoint, Keypoint>{ leftEar, rightEar }
            };
            for (const auto& pair : pairs) {
                if (!pair.first.visible || !pair.second.visible) continue;
                const float dx = pair.first.point.x - pair.second.point.x;
                if (std::abs(dx) < 1e-6f) continue;
                const float t = (nose.point.x - pair.second.point.x) / dx;
                cues.push_back(std::clamp(1.0f - 2.0f * std::abs(t - 0.5f), 0.0f, 1.0f));
            }
            float cueScore = 0.0f;
            if (!cues.empty()) {
                for (float cue : cues) cueScore += cue;
                cueScore /= static_cast<float>(cues.size());
            }
            faceScore = std::max(0.20f, cueScore);
        }
        else if (leftEye.visible || rightEye.visible) {
            faceScore = 0.20f;
        }
        else if (leftEar.visible || rightEar.visible) {
            faceScore = 0.0f;
        }
        const bool faceVisible = nose.visible || (leftEye.visible && rightEye.visible);
        const bool shoulderOrderCheckApplies = leftShoulder.visible && rightShoulder.visible &&
            std::abs(leftShoulder.point.x - rightShoulder.point.x) > 0.15f * person.box.width;
        const bool shoulderOrderIndicatesBack = shoulderOrderCheckApplies &&
            leftShoulder.point.x - rightShoulder.point.x < 0.0f;
        if (shoulderOrderIndicatesBack ||
            (!shoulderOrderCheckApplies && !faceVisible && (leftEar.visible || rightEar.visible))) {
            regions.view = "back";
        }
        else if (bodyScore.has_value()) {
            regions.view = *bodyScore >= kFrontThreshold ? "front" : "side";
        }
        else if (faceScore >= 0.0f) {
            regions.view = faceScore >= kFrontThreshold ? "front" : "side";
        }
        regions.bodyFacing = bodyScore;
        if (faceScore >= 0.0f) regions.faceFacing = faceScore;
        return regions;
    }

    cv::Mat applyGrayWorldBalance(const cv::Mat& frame) {
        if (frame.empty()) return {};
        cv::Scalar means = cv::mean(frame);
        const double gray = (means[0] + means[1] + means[2]) / 3.0;
        cv::Mat balanced;
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        for (size_t channel = 0; channel < channels.size(); ++channel) {
            const double gain = gray / std::max(means[static_cast<int>(channel)], 1e-3);
            channels[channel].convertTo(channels[channel], CV_8U, gain);
        }
        cv::merge(channels, balanced);
        return balanced;
    }

    int classifyAllowedColor(const cv::Mat& frame, const ANSCENTER::Object& detection,
        int itemIndex, std::string* detectedColor = nullptr) {
        if (detectedColor) *detectedColor = "UNKNOWN";
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
            }
            else {
                cv::resize(mask, itemMask, box.size(), 0.0, 0.0, cv::INTER_NEAREST);
            }
            cv::threshold(itemMask, itemMask, 0.5, 255.0, cv::THRESH_BINARY);
            itemMask.convertTo(itemMask, CV_8U);
            if (cv::countNonZero(itemMask) > 0) {
                cv::Mat eroded;
                const cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::erode(itemMask, eroded, kernel);
                if (cv::countNonZero(eroded) >= 50) itemMask = std::move(eroded);
            }
        }
        else {
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

        int whitePixels = 0;
        int redPixels = 0;
        int tintedWhitePixels = 0;
        const bool tintedWhiteApplies = itemIndex == 1;
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
                const bool tintedWhite = tintedWhiteApplies && h >= 90 && h <= 135 &&
                    s <= 110 && v >= 60 && v <= 190;
                if (red) ++redPixels;
                else if (white) ++whitePixels;
                else if (tintedWhite) {
                    ++whitePixels;
                    ++tintedWhitePixels;
                }
            }
        }
        const int totalPixels = cv::countNonZero(itemMask);
        if (totalPixels < 50) return -1;
        const bool allowed = static_cast<float>(redPixels + whitePixels) / totalPixels >= 0.70f;
        if (detectedColor) {
            *detectedColor = allowed ? (redPixels > whitePixels ? "RED" : "WHITE") : "OTHER";
            if (allowed && redPixels <= whitePixels && tintedWhitePixels * 2 > whitePixels) {
                *detectedColor = "WHITE_NET";
            }
        }
        return allowed ? 1 : 0;
    }

    void drawDebugLabel(cv::Mat& frame, const std::string& text, cv::Point origin,
        const cv::Scalar& color, double scale = 0.5) {
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
        origin.x = std::clamp(origin.x, 0, std::max(0, frame.cols - textSize.width - 4));
        origin.y = std::clamp(origin.y, textSize.height + baseline + 4, std::max(
            textSize.height + baseline + 4, frame.rows - 1));
        cv::rectangle(frame, cv::Point(origin.x, origin.y - textSize.height - baseline - 4),
            cv::Point(origin.x + textSize.width + 4, origin.y + 2), color, cv::FILLED);
        cv::putText(frame, text, cv::Point(origin.x + 2, origin.y - baseline - 1),
            cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    void drawDetectionMask(cv::Mat& frame, const ANSCENTER::Object& detection,
        const cv::Scalar& color, double alpha) {
        if (detection.box.area() <= 0 || detection.mask.empty()) return;
        const cv::Rect frameBounds(0, 0, frame.cols, frame.rows);
        const cv::Rect clippedBox = detection.box & frameBounds;
        if (clippedBox.empty()) return;

        cv::Mat maskSource = detection.mask;
        if (maskSource.channels() > 1) {
            cv::cvtColor(maskSource, maskSource, cv::COLOR_BGR2GRAY);
        }
        cv::Mat mask;
        if (detection.mask.size() == frame.size()) {
            mask = maskSource(clippedBox);
        }
        else {
            cv::Mat localMask = maskSource;
            cv::resize(localMask, localMask, detection.box.size(), 0.0, 0.0, cv::INTER_NEAREST);
            const cv::Rect localBox(
                clippedBox.x - detection.box.x, clippedBox.y - detection.box.y,
                clippedBox.width, clippedBox.height);
            if (localBox.x < 0 || localBox.y < 0 ||
                localBox.x + localBox.width > localMask.cols ||
                localBox.y + localBox.height > localMask.rows) return;
            mask = localMask(localBox);
        }
        cv::Mat binaryMask;
        cv::compare(mask, 0, binaryMask, cv::CMP_GT);
        cv::Mat roi = frame(clippedBox);
        cv::Mat tinted(roi.size(), roi.type(), color);
        cv::Mat blended;
        cv::addWeighted(roi, 1.0 - alpha, tinted, alpha, 0.0, blended);
        blended.copyTo(roi, binaryMask);
    }

    void drawPoseDebug(cv::Mat& frame, const ANSCENTER::Object& person) {
        constexpr float confidenceThreshold = 0.40f;
        constexpr std::array<std::pair<int, int>, 12> skeleton{ {
            {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7},
            {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}
        } };
        const auto point = [&](int index, cv::Point& output) {
            const size_t offset = static_cast<size_t>(index) * 3;
            if (person.kps.size() < offset + 3 || person.kps[offset + 2] < confidenceThreshold ||
                !std::isfinite(person.kps[offset]) || !std::isfinite(person.kps[offset + 1])) return false;
            output = cv::Point(cvRound(person.kps[offset]), cvRound(person.kps[offset + 1]));
            return true;
            };
        for (const auto& edge : skeleton) {
            cv::Point first, second;
            if (point(edge.first, first) && point(edge.second, second)) {
                cv::line(frame, first, second, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
            }
        }
        for (size_t index = 0; index + 2 < person.kps.size(); index += 3) {
            if (person.kps[index + 2] < confidenceThreshold ||
                !std::isfinite(person.kps[index]) || !std::isfinite(person.kps[index + 1])) continue;
            cv::circle(frame, cv::Point(cvRound(person.kps[index]), cvRound(person.kps[index + 1])),
                3, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
        }
    }

}

ANSCustomTL::ANSCustomTL() = default;

bool ANSCustomTL::OptimizeModel(bool fp16) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    if (!_uniformDetector || !_uniformDetector->IsInitialized() || !_poseDetector) {
        std::cerr << "[ANSKitchenUniform] Cannot optimize before model initialization." << std::endl;
        return false;
    }
    const bool poseOptimized = _poseDetector->Optimize(fp16);
    std::cout << "[ANSKitchenUniform] RF-DETR is running through ONNX Runtime; "
        << "ANSLIB TensorRT optimization does not apply to the uniform model." << std::endl;
    if (!poseOptimized) {
        std::cerr << "[ANSKitchenUniform] Pose model optimization failed." << std::endl;
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
    _inferenceFrameIndex = 0;
    _debugFrame.release();
    _debugPersonFrames.clear();
    _readParam = false;
    return true;
}

void ANSCustomTL::SetDebugVisualization(bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _debugVisualizationEnabled = enabled;
    if (!enabled) {
        _debugFrame.release();
        _debugPersonFrames.clear();
    }
}

bool ANSCustomTL::GetDebugFrame(cv::Mat& frame) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    if (_debugFrame.empty()) return false;
    frame = _debugFrame.clone();
    return true;
}

bool ANSCustomTL::GetDebugPersonFrames(std::vector<UniformDebugPersonFrame>& frames) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    frames.clear();
    frames.reserve(_debugPersonFrames.size());
    for (const auto& personFrame : _debugPersonFrames) {
        UniformDebugPersonFrame copy = personFrame;
        copy.annotatedCrop = personFrame.annotatedCrop.clone();
        frames.push_back(std::move(copy));
    }
    return !frames.empty();
}

void ANSCustomTL::GetDebugTimings(
    double& poseMilliseconds, double& uniformMilliseconds, bool& uniformRan) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    poseMilliseconds = _debugPoseInferenceMilliseconds;
    uniformMilliseconds = _debugUniformInferenceMilliseconds;
    uniformRan = _debugUniformInferenceRan;
}

bool ANSCustomTL::Initialize(const std::string& modelDirectory, float detectionScoreThreshold, std::string& labelMap) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    try {
        (void)detectionScoreThreshold;
        _modelDirectory = modelDirectory;
        _detectionScoreThreshold = 0.25f;
        _uniformModelName = "uniform";
        _poseModelName = "pose";
        _poseClassName = "pose.names";
        _poseModelType = 31; // ONNXPOSE
        _poseDetectionType = 7; // KEYPOINT
        _uniformDetector = std::make_unique<DetrUniformDetector>();
        _poseDetector = std::make_unique<ANSCENTER::ANSLIB>();

        constexpr float poseModelConfidence = 0.25f;
        constexpr float poseNmsThreshold = 0.50f;
        constexpr int autoEngineDetection = 1;
        //#ifdef FNS_DEBUG
        _loadEngineOnCreate = true;
        //#endif
        const int loadEngineOnCreation = _loadEngineOnCreate ? 1 : 0;
        const std::string licenseKey;

        const std::filesystem::path modelDirectoryPath(_modelDirectory);
        const bool uniformResult = _uniformDetector->Initialize(
            (modelDirectoryPath / "uniform.onnx").string(),
            (modelDirectoryPath / "uniform.names").string(),
            0.25f);
        const int poseResult = _poseDetector->LoadModelFromFolder(
            licenseKey.c_str(), _poseModelName.c_str(), _poseClassName.c_str(),
            _detectionScoreThreshold, poseModelConfidence, poseNmsThreshold,
            autoEngineDetection, _poseModelType, _poseDetectionType, loadEngineOnCreation,
            _modelDirectory.c_str(), _poseLabelMap);
        if (!uniformResult || poseResult != 1) {
            std::cerr << "[ANSKitchenUniform] Model loading failed: uniform="
                << (uniformResult ? 1 : 0)
                << ", pose=" << poseResult << std::endl;
            Destroy();
            return false;
        }

        const int poseTrackerResult = _poseDetector->SetTracker(BYTETRACK, 1);
        if (poseTrackerResult != 1) {
            std::cerr << "[ANSKitchenUniform] Pose tracker initialization warning: "
                << poseTrackerResult << std::endl;
        }

        labelMap = "FaceMask,ChefHat,KitchenGloves,Apron,Non_Compliant";
        _uniformTracks.clear();
        return true;
    }
    catch (const std::exception& error) {
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
    ++_inferenceFrameIndex;
    if (!_uniformDetector || !_uniformDetector->IsInitialized() || !_poseDetector) {
        std::cerr << "[ANSKitchenUniform] RunInference called before model initialization." << std::endl;
        return results;
    }

    if (_debugVisualizationEnabled) {
        _debugFrame = input.clone();
        _debugPersonFrames.clear();
        _debugPoseInferenceMilliseconds = 0.0;
        _debugUniformInferenceMilliseconds = 0.0;
        _debugUniformInferenceRan = false;
    }
    std::vector<ANSCENTER::Object> people;
    std::vector<std::vector<ANSCENTER::Object>> detectionsByPerson;
    std::vector<UniformTrackState*> trackStates;
    const auto renderDebugFrame = [&](const std::string& message) {
        if (!_debugVisualizationEnabled || _debugFrame.empty()) return;

        int compliantCount = 0;
        int violationCount = 0;
        int undeterminedCount = 0;
        for (size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
            const auto& person = people[personIndex];
            const auto stateIt = _uniformTracks.find(person.trackId);
            const UniformTrackState* state = stateIt == _uniformTracks.end() ? nullptr : &stateIt->second;
            const PersonRegions regions = assessPerson(person, _debugFrame.cols, _debugFrame.rows);
            if (personIndex < detectionsByPerson.size()) {
                for (const auto& detection : detectionsByPerson[personIndex]) {
                    const int itemIndex = detectionItemIndex(detection);
                    const cv::Point center(
                        detection.box.x + detection.box.width / 2,
                        detection.box.y + detection.box.height / 2);
                    if (itemIndex < 0 || !person.box.contains(center) ||
                        !detectionMatchesPoseRegion(itemIndex, center, regions)) continue;
                    const cv::Scalar color = kItemColors[static_cast<size_t>(itemIndex)];
                    drawDetectionMask(_debugFrame, detection, color, 0.45);
                    cv::rectangle(_debugFrame, detection.box, color, 2, cv::LINE_AA);
                    std::ostringstream label;
                    label << detection.className << " " << std::fixed << std::setprecision(2)
                        << detection.confidence;
                    drawDebugLabel(_debugFrame, label.str(),
                        cv::Point(detection.box.x, detection.box.y - 4), color);
                }
            }
            bool hasViolation = false;
            if (state) {
                for (const auto& status : state->debugItemStatuses) {
                    hasViolation = hasViolation || status == "MISSING" || status == "WRONG_COLOR";
                }
            }
            const bool compliant = state && state->compliant;
            const std::string verdict = compliant ? "COMPLIANT"
                : hasViolation ? "NON_COMPLIANT" : "UNDETERMINED";
            const cv::Scalar verdictColor = compliant ? cv::Scalar(0, 200, 0)
                : hasViolation ? cv::Scalar(0, 0, 230) : cv::Scalar(150, 150, 150);
            if (compliant) ++compliantCount;
            else if (hasViolation) ++violationCount;
            else ++undeterminedCount;

            const auto drawRegion = [&](const cv::Rect2f& box, bool visible, const cv::Scalar& visibleColor) {
                if (box.width <= 0.0f || box.height <= 0.0f) return;
                cv::rectangle(_debugFrame, box, visible ? visibleColor : cv::Scalar(90, 90, 90),
                    1, cv::LINE_AA);
                };
            drawRegion(regions.headBox, regions.head, cv::Scalar(200, 200, 200));
            drawRegion(regions.crownBox, regions.head, cv::Scalar(255, 255, 0));
            drawRegion(regions.torsoBox, regions.torso, cv::Scalar(200, 200, 200));
            drawRegion(regions.leftHandBox, regions.leftHand, cv::Scalar(255, 255, 0));
            drawRegion(regions.rightHandBox, regions.rightHand, cv::Scalar(255, 255, 0));
            drawPoseDebug(_debugFrame, person);
            cv::rectangle(_debugFrame, person.box, verdictColor, 2, cv::LINE_AA);
            const std::string id = person.trackId >= 0
                ? "ID " + std::to_string(person.trackId) : "PERSON";
            std::ostringstream orientation;
            orientation << regions.view;
            if (regions.bodyFacing) {
                orientation << " b" << std::fixed << std::setprecision(2) << *regions.bodyFacing;
            }
            if (regions.faceFacing) {
                orientation << " f" << std::fixed << std::setprecision(2) << *regions.faceFacing;
            }
            const std::string alert = state && state->alertRaised ? " ALERT" : "";
            drawDebugLabel(_debugFrame, id + " " + verdict + " [" + orientation.str() + "]" + alert,
                cv::Point(person.box.x, person.box.y - 5), verdictColor);

            int labelY = person.box.y + 18;
            for (size_t item = 0; item < kItemNames.size(); ++item) {
                std::string status = state ? state->debugItemStatuses[item] : std::string();
                if (status.empty()) status = "NOT CHECKED";
                const std::string color = state ? state->debugItemColors[item] : std::string();
                const std::string detail = color.empty() ? std::string() : " (" + color + ")";
                const cv::Scalar statusColor = status == "OK" ? cv::Scalar(0, 200, 0)
                    : status == "MISSING" || status == "WRONG_COLOR" ? cv::Scalar(0, 0, 230)
                    : status == "DISABLED" ? cv::Scalar(90, 90, 90)
                    : cv::Scalar(150, 150, 150);
                drawDebugLabel(_debugFrame, std::string(kItemNames[item]) + ": " + status + detail,
                    cv::Point(person.box.x + 2, labelY), statusColor, 0.42);
                labelY += 20;
            }
        }

        _debugPersonFrames.clear();
        _debugPersonFrames.reserve(people.size());
        for (size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
            const auto& person = people[personIndex];
            const cv::Rect personBox = person.box & cv::Rect(0, 0, _debugFrame.cols, _debugFrame.rows);
            if (personBox.empty()) continue;

            UniformDebugPersonFrame personFrame;
            personFrame.trackId = person.trackId;
            personFrame.box = personBox;
            const auto stateIt = _uniformTracks.find(person.trackId);
            const UniformTrackState* state = stateIt == _uniformTracks.end() ? nullptr : &stateIt->second;
            if (state) {
                personFrame.itemStatuses = state->debugItemStatuses;
                bool hasViolation = false;
                for (const auto& status : state->debugItemStatuses) {
                    hasViolation = hasViolation || status == "MISSING" || status == "WRONG_COLOR";
                }
                personFrame.verdict = state->compliant ? "COMPLIANT"
                    : hasViolation ? "NON_COMPLIANT" : "UNDETERMINED";
            }
            else {
                personFrame.verdict = "UNDETERMINED";
            }
            if (personIndex < detectionsByPerson.size()) {
                const PersonRegions regions = assessPerson(person, _debugFrame.cols, _debugFrame.rows);
                for (const auto& detection : detectionsByPerson[personIndex]) {
                    const int itemIndex = detectionItemIndex(detection);
                    const cv::Point center(
                        detection.box.x + detection.box.width / 2,
                        detection.box.y + detection.box.height / 2);
                    if (itemIndex < 0 || !personBox.contains(center) ||
                        !detectionMatchesPoseRegion(itemIndex, center, regions)) continue;
                    std::ostringstream detail;
                    detail << detection.className << " " << std::fixed << std::setprecision(3)
                        << detection.confidence << " box=(" << detection.box.x - personBox.x
                        << "," << detection.box.y - personBox.y << ","
                        << detection.box.width << "," << detection.box.height << ")";
                    personFrame.detections.push_back(detail.str());
                }
            }
            personFrame.annotatedCrop = _debugFrame(personBox).clone();
            _debugPersonFrames.push_back(std::move(personFrame));
        }

        std::ostringstream banner;
        banner << "Persons " << people.size() << " | OK " << compliantCount
            << " | VIOLATION " << violationCount << " | UNDET " << undeterminedCount;
        if (!message.empty()) banner << " | " << message;
        drawDebugLabel(_debugFrame, banner.str(), cv::Point(5, 22), cv::Scalar(35, 35, 35), 0.55);
        };

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
                }
                catch (const std::exception&) {
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
        const auto poseStartTime = std::chrono::steady_clock::now();
        const int poseStatus = _poseDetector->RunInference(input, camera_id.c_str(), people);
        if (_debugVisualizationEnabled) {
            _debugPoseInferenceMilliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - poseStartTime).count();
        }
        if (poseStatus != 1) {
            std::cerr << "[ANSKitchenUniform] Pose inference failed with status " << poseStatus << "." << std::endl;
            renderDebugFrame("POSE INFERENCE FAILED");
            return results;
        }
        const size_t detectedPersonCount = people.size();
        people.erase(std::remove_if(people.begin(), people.end(), [&](const ANSCENTER::Object& person) {
            const cv::Rect box = person.box & cv::Rect(0, 0, input.cols, input.rows);
            return box.area() < kMinPersonBoxArea ||
                static_cast<float>(box.height) < kMinPersonHeightRatio * input.rows;
            }), people.end());
        if (people.size() != detectedPersonCount) {
            std::cout << "[ANSKitchenUniform] Filtered "
                << (detectedPersonCount - people.size())
                << " pose person(s): box area < " << kMinPersonBoxArea
                << " px or height < " << kMinPersonHeightRatio * 100.0f
                << "% of frame." << std::endl;
        }
        if (people.empty()) {
            for (auto it = _uniformTracks.begin(); it != _uniformTracks.end();) {
                it->second.wasFrontMissing = false;
                if (it->second.hasSeen && currentTime - it->second.lastSeenTime > keepTrackDuration) {
                    it = _uniformTracks.erase(it);
                }
                else ++it;
            }
            renderDebugFrame("NO PERSON");
            return results;
        }

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
            }
            else {
                ++it;
            }
        }

        if (!shouldRunUniform) {
            renderDebugFrame("uniform check cached");
            return results;
        }

        const cv::Rect frameBounds(0, 0, input.cols, input.rows);
        _debugUniformInferenceRan = _debugVisualizationEnabled;
        detectionsByPerson.resize(people.size());
        for (size_t personIndex = 0; personIndex < people.size(); ++personIndex) {
            const auto& person = people[personIndex];
            const cv::Rect personCropBox = person.box & frameBounds;
            if (personCropBox.empty()) continue;

            const cv::Mat personCrop = input(personCropBox);
            const auto uniformStartTime = std::chrono::steady_clock::now();
            const int uniformStatus = _uniformDetector->RunInference(
                personCrop, camera_id.c_str(), detectionsByPerson[personIndex]);
            if (_debugVisualizationEnabled) {
                _debugUniformInferenceMilliseconds += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - uniformStartTime).count();
            }
            if (uniformStatus != 1) {
                std::cerr << "[ANSKitchenUniform] Uniform inference failed for track "
                    << person.trackId << " with status " << uniformStatus << "." << std::endl;
                renderDebugFrame("UNIFORM INFERENCE FAILED");
                return results;
            }

            for (auto& detection : detectionsByPerson[personIndex]) {
                detection.box.x += personCropBox.x;
                detection.box.y += personCropBox.y;
            }
        }

        const cv::Mat colorAnalysisFrame = applyGrayWorldBalance(input);
        const auto candidateDetections = [&](int wantedItem, const cv::Rect& personBox,
            const PersonRegions& regions,
            const std::vector<ANSCENTER::Object>& personDetections) {
                std::vector<const ANSCENTER::Object*> candidates;
                for (const auto& detection : personDetections) {
                    const int detectedItem = detectionItemIndex(detection);
                    if (detectedItem != wantedItem || detection.box.area() <= 0 || personBox.area() <= 0) continue;
                    const cv::Point center(
                        detection.box.x + detection.box.width / 2,
                        detection.box.y + detection.box.height / 2);
                    if (!personBox.contains(center) ||
                        !detectionMatchesPoseRegion(detectedItem, center, regions)) continue;
                    candidates.push_back(&detection);
                }
                std::sort(candidates.begin(), candidates.end(),
                    [](const ANSCENTER::Object* left, const ANSCENTER::Object* right) {
                        return left->confidence > right->confidence;
                    });
                return candidates;
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
                if (!_requiredItems[item]) {
                    state.debugItemStatuses[item] = "DISABLED";
                    continue;
                }
                const bool itemVisible = item == 0 ? regions.face
                    : item == 1 ? regions.head
                    : item == 2 ? (regions.leftHand || regions.rightHand)
                    : regions.torso;
                const auto candidates = candidateDetections(static_cast<int>(item), person.box,
                    regions, detectionsByPerson[personIndex]);
                const size_t maxColorCandidates = item == 3 ? 3u : 1u;
                const size_t candidateLimit = (item == 1 || item == 3)
                    ? std::min(maxColorCandidates, candidates.size())
                    : std::min<size_t>(1, candidates.size());
                const ANSCENTER::Object* detection = nullptr;
                int colorStatus = 1;
                std::string detectedColor;
                for (size_t candidateIndex = 0; candidateIndex < candidateLimit; ++candidateIndex) {
                    const ANSCENTER::Object* candidate = candidates[candidateIndex];
                    const int candidateColorStatus = (item == 1 || item == 3)
                        ? classifyAllowedColor(colorAnalysisFrame, *candidate,
                            static_cast<int>(item), nullptr)
                        : 1;
                    if (detection == nullptr) {
                        detection = candidate;
                        colorStatus = candidateColorStatus;
                    }
                    if (candidateColorStatus == 1) {
                        detection = candidate;
                        colorStatus = candidateColorStatus;
                        break;
                    }
                }
                const bool found = detection != nullptr;
                if (found && (item == 1 || item == 3) && colorStatus >= 0) {
                    classifyAllowedColor(colorAnalysisFrame, *detection,
                        static_cast<int>(item), &detectedColor);
                }
                else if (found && colorStatus < 0) {
                    detectedColor = "UNKNOWN";
                }

                int itemStatus = !itemVisible || (found && colorStatus < 0)
                    ? 0
                    : !found ? 1
                    : colorStatus == 0 ? 2 : 3;
                if (itemStatus == 3) {
                    state.lastPositiveFrame[item] = _inferenceFrameIndex;
                }
                else if (itemStatus == 1 && state.lastPositiveFrame[item] >= 0 &&
                    _inferenceFrameIndex - state.lastPositiveFrame[item] <= 60) {
                    itemStatus = 3;
                }

                auto& history = state.itemStatusHistory[item];
                history.push_back(itemStatus);
                if (history.size() > 5) history.pop_front();
                int smoothedStatus = itemStatus;
                if (itemStatus != 0) {
                    std::array<int, 4> counts{};
                    for (const int priorStatus : history) {
                        if (priorStatus != 0) ++counts[static_cast<size_t>(priorStatus)];
                    }
                    int winningCount = -1;
                    for (auto it = history.rbegin(); it != history.rend(); ++it) {
                        if (*it == 0) continue;
                        const int count = counts[static_cast<size_t>(*it)];
                        if (count > winningCount) {
                            winningCount = count;
                            smoothedStatus = *it;
                        }
                    }
                }

                static constexpr std::array<const char*, 4> statusNames{
                    "OCCLUDED", "MISSING", "WRONG_COLOR", "OK"
                };
                state.debugItemStatuses[item] = statusNames[static_cast<size_t>(smoothedStatus)];
                state.debugItemColors[item] = std::move(detectedColor);

                if (smoothedStatus == 0) {
                    continue;
                }
                if (smoothedStatus == 3) {
                    state.confirmedItems.insert(static_cast<int>(item));
                }
                else if (smoothedStatus == 2) {
                    state.confirmedItems.erase(static_cast<int>(item));
                    visibleViolation = true;
                }
                else if (smoothedStatus == 1) {
                    state.confirmedItems.erase(static_cast<int>(item));
                    visibleViolation = true;
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

        renderDebugFrame("live");
    }
    catch (const std::exception& error) {
        std::cerr << "[ANSKitchenUniform] Inference error: " << error.what() << std::endl;
        renderDebugFrame("INFERENCE ERROR");
    }
    return results;
}
