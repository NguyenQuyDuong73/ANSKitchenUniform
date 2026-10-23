#include <ANSKitchenUniform.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <windows.h>
#include <nlohmann/json.hpp>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace {
    using Json = nlohmann::json;

    bool LoadParametersFromJSON(const std::string& configFilePath, CustomParams& customParams) {
        std::ifstream configFile(configFilePath);
        if (!configFile.is_open()) {
            std::cerr << "Could not open parameter template " << configFilePath
                << "; retaining default parameters." << std::endl;
            return false;
        }

        try {
            const Json config = Json::parse(
                std::string(std::istreambuf_iterator<char>(configFile), std::istreambuf_iterator<char>()));
            if (!config.contains("Parameters") || !config["Parameters"].is_array()) {
                std::cerr << "Parameter template has no Parameters array."
                    << std::endl;
                return false;
            }

            CustomParams configuredParams = customParams;
            const auto readString = [](const Json& value) {
                if (value.is_string()) return value.get<std::string>();
                if (value.is_boolean()) return value.get<bool>() ? std::string("true") : std::string("false");
                if (value.is_number()) return value.dump();
                return std::string();
                };
            for (const auto& configuredParam : config["Parameters"]) {
                if (!configuredParam.is_object() || !configuredParam.contains("Name") ||
                    !configuredParam["Name"].is_string()) continue;
                const std::string name = configuredParam["Name"].get<std::string>();
                auto target = std::find_if(configuredParams.Parameters.begin(), configuredParams.Parameters.end(),
                    [&](const CustomParameter& param) { return param.Name == name; });
                if (target == configuredParams.Parameters.end()) continue;

                if (configuredParam.contains("DefaultValue")) {
                    target->DefaultValue = readString(configuredParam["DefaultValue"]);
                }
                if (configuredParam.contains("StartValue")) {
                    target->StartValue = readString(configuredParam["StartValue"]);
                }
                if (configuredParam.contains("Value")) {
                    target->Value = readString(configuredParam["Value"]);
                }
            }
            customParams = std::move(configuredParams);
            std::cout << "Loaded parameter template: " << configFilePath << std::endl;
            return true;
        }
        catch (const std::exception& error) {
            std::cerr << "Failed to parse parameter template " << configFilePath
                << ": " << error.what() << "; retaining default parameters." << std::endl;
            return false;
        }
    }
}

int main(int argc, char** argv) {
    SetDllDirectoryW(L"C:\\ProgramData\\ANSCENTER\\Shared");

    const std::string assetDirectory = argc > 1
        ? argv[1]
        : "C:\\Users\\ADMIN\\Downloads\\vid_namdu\\ANS_KitchenUniform(GPU)_1";
    const std::string videoFilePath = argc > 2
        ? argv[2]
        : assetDirectory + "\\test1.mp4";
    const int maxFrames = argc > 3 ? std::max(0, std::stoi(argv[3])) : 0;
    const std::string configFilePath = argc > 4 ? argv[4] : "config.json";
    const std::filesystem::path outputDirectory = argc > 5
        ? std::filesystem::path(argv[5])
        : std::filesystem::path(assetDirectory) / "output";
    const std::filesystem::path alertsDirectory = outputDirectory / "alerts";
    try {
        std::filesystem::create_directories(alertsDirectory);
    }
    catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "Could not create output folders under " << outputDirectory.string()
            << ": " << error.what() << std::endl;
        return 1;
    }
    std::ofstream runLog(outputDirectory / "run.log", std::ios::out | std::ios::trunc);
    if (!runLog.is_open()) {
        std::cerr << "Could not create log file: " << (outputDirectory / "run.log").string() << std::endl;
        return 1;
    }
    const auto log = [&](const std::string& message) {
        std::cout << message << std::endl;
        runLog << message << std::endl;
        };
    log("[INIT] video=" + videoFilePath + " | output=" + outputDirectory.string());

    ANSCustomTL uniformChecker;
    CustomParams customParams;
    if (!uniformChecker.ConfigureParameters(customParams)) {
        std::cerr << "ConfigureParameters failed." << std::endl;
        return 1;
    }

    const std::vector<std::string> expectedNames{
        "ChefHat", "FaceMask", "KitchenGloves", "Apron", "CheckWrongColor",
        "AlertAfterSecond", "CompliantRecheck", "KeepTrackObj"
    };
    if (customParams.Parameters.size() != expectedNames.size()) {
        std::cerr << "Unexpected parameter count: " << customParams.Parameters.size() << std::endl;
        return 1;
    }
    for (size_t index = 0; index < expectedNames.size(); ++index) {
        const auto& param = customParams.Parameters[index];
        if (param.Name != expectedNames[index]) {
            std::cerr << "Unexpected parameter at position " << index << ": " << param.Name << std::endl;
            return 1;
        }
        if (index < 4 && param.DefaultValue != "true") {
            std::cerr << "Uniform requirement is not enabled by default: " << param.Name << std::endl;
            return 1;
        }
    }
    if (customParams.Parameters[4].DefaultValue != "true" ||
        customParams.Parameters[5].DefaultValue != "5" ||
        customParams.Parameters[6].DefaultValue != "60" ||
        customParams.Parameters[7].DefaultValue != "5") {
        std::cerr << "Unexpected alert/recheck/track-retention defaults." << std::endl;
        return 1;
    }
    if (!LoadParametersFromJSON(configFilePath, customParams)) {
        std::cout << "Using default uniform parameters." << std::endl;
    }
    if (!uniformChecker.SetParameters(customParams)) {
        std::cerr << "SetParameters failed." << std::endl;
        return 1;
    }
    uniformChecker.SetDebugVisualization(true);

    std::string labelMap;
    if (!uniformChecker.Initialize(assetDirectory, 0.25f, labelMap)) {
        std::cerr << "Failed to initialize uniform and pose ONNX models from "
            << assetDirectory << std::endl;
        log("[ERROR] Model initialization failed. Check pose license and model files.");
        return 1;
    }
    log("[INIT] Models initialized | event labels=" + labelMap);

    cv::VideoCapture capture(videoFilePath);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open test video: " << videoFilePath << std::endl;
        log("[ERROR] Failed to open input video.");
        return 1;
    }
    log("[INIT] Video opened | size=" +
        std::to_string(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH))) + "x" +
        std::to_string(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT))) +
        " | fps=" + std::to_string(capture.get(cv::CAP_PROP_FPS)));

    const int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    const int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    cv::namedWindow("FullScreen", cv::WINDOW_NORMAL);
    cv::setWindowProperty("FullScreen", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    cv::Mat frame;
    int frameIndex = 0;
    bool outputWriteFailed = false;
    const double startTick = static_cast<double>(cv::getTickCount());
    while (capture.read(frame)) {
        const double inferenceStartTick = static_cast<double>(cv::getTickCount());
        const std::vector<CustomObject> events = uniformChecker.RunInference(frame, "KitchenUniformTest");
        const double inferenceMilliseconds =
            (static_cast<double>(cv::getTickCount()) - inferenceStartTick) /
            cv::getTickFrequency() * 1000.0;
        std::vector<UniformDebugPersonFrame> personFrames;
        uniformChecker.GetDebugPersonFrames(personFrames);
        double poseMilliseconds = 0.0;
        double uniformMilliseconds = 0.0;
        bool uniformInferenceRan = false;
        uniformChecker.GetDebugTimings(
            poseMilliseconds, uniformMilliseconds, uniformInferenceRan);
        size_t detectionCount = 0;
        for (const auto& personFrame : personFrames) detectionCount += personFrame.detections.size();
        {
            std::ostringstream frameLog;
            frameLog << "[FRAME] index=" << frameIndex << " pose_persons=" << personFrames.size()
                << " uniform_objects=" << detectionCount << " events=" << events.size()
                << " pose_ms=" << std::fixed << std::setprecision(1) << poseMilliseconds
                << " uniform_ms=" << uniformMilliseconds
                << " uniform_mode=" << (uniformInferenceRan ? "inference" : "cached")
                << " total_ms=" << inferenceMilliseconds;
            log(frameLog.str());
        }

        cv::Mat annotated;
        if (!uniformChecker.GetDebugFrame(annotated)) annotated = frame.clone();
        const double elapsed = static_cast<double>(cv::getTickCount() - startTick) /
            cv::getTickFrequency();
        const double fps = (frameIndex + 1) / std::max(elapsed, 1e-6);
        const std::string performance = "Frame " + std::to_string(frameIndex) +
            " | " + std::to_string(static_cast<int>(fps + 0.5)) + " FPS";
        cv::putText(annotated, performance, cv::Point(8, annotated.rows - 12),
            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(annotated, performance, cv::Point(8, annotated.rows - 12),
            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        for (const auto& personFrame : personFrames) {
            std::ostringstream personLog;
            personLog << "[PERSON] frame=" << frameIndex << " track=" << personFrame.trackId
                << " verdict=" << personFrame.verdict << " bbox=(" << personFrame.box.x << ","
                << personFrame.box.y << "," << personFrame.box.width << "," << personFrame.box.height
                << ") statuses=";
            for (size_t item = 0; item < personFrame.itemStatuses.size(); ++item) {
                if (item > 0) personLog << ",";
                personLog << personFrame.itemStatuses[item];
            }
            personLog << " detections=" << personFrame.detections.size()
                << " (diagnostic only; snapshots are saved for emitted violations)";
            log(personLog.str());
            for (const auto& detection : personFrame.detections) {
                log("[OBJECT] frame=" + std::to_string(frameIndex) + " track=" +
                    std::to_string(personFrame.trackId) + " " + detection);
            }
        }

        for (const auto& event : events) {
            std::ostringstream eventLog;
            eventLog << "[EVENT] frame=" << frameIndex << " class=" << event.className
                << " track=" << event.trackId << " confidence=" << std::fixed
                << std::setprecision(3) << event.confidence;
            if (!event.extraInfo.empty()) eventLog << ' ' << event.extraInfo;
            const auto personFrame = std::find_if(personFrames.begin(), personFrames.end(),
                [&](const UniformDebugPersonFrame& candidate) {
                    return candidate.trackId == event.trackId;
                });
            if (personFrame == personFrames.end() || personFrame->annotatedCrop.empty()) {
                log("[ERROR] Cannot save violation snapshot: debug crop missing for track " +
                    std::to_string(event.trackId));
                outputWriteFailed = true;
                log(eventLog.str());
                continue;
            }
            std::ostringstream alertFileName;
            alertFileName << "frame_" << std::setw(6) << std::setfill('0') << frameIndex
                << "_track_" << event.trackId << ".jpg";
            const std::filesystem::path alertPath = alertsDirectory / alertFileName.str();
            if (!cv::imwrite(alertPath.string(), personFrame->annotatedCrop)) {
                log("[ERROR] Failed to write violation snapshot: " + alertPath.string());
                outputWriteFailed = true;
            }
            else {
                eventLog << " snapshot=" << alertPath.string();
            }
            log(eventLog.str());
        }

        const double scale = std::min(
            static_cast<double>(screenWidth) / annotated.cols,
            static_cast<double>(screenHeight) / annotated.rows);
        cv::Mat resized;
        cv::resize(annotated, resized, cv::Size(), scale, scale);
        cv::Mat display(screenHeight, screenWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        const int x = (screenWidth - resized.cols) / 2;
        const int y = (screenHeight - resized.rows) / 2;
        resized.copyTo(display(cv::Rect(x, y, resized.cols, resized.rows)));
        cv::imshow("FullScreen", display);
        ++frameIndex;
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || (maxFrames > 0 && frameIndex >= maxFrames)) break;
    }

    capture.release();
    cv::destroyAllWindows();
    uniformChecker.Destroy();
    log("[DONE] frames=" + std::to_string(frameIndex) +
        " | violation snapshots=" + alertsDirectory.string());
    runLog.close();
    return outputWriteFailed ? 1 : 0;
}
