#include <ANSKitchenUniform.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
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
        } catch (const std::exception& error) {
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
        : "C:\\Users\\Admin\\Downloads\\ANS_KitchenUniform(GPU)_1";
    const std::string videoFilePath = argc > 2
        ? argv[2]
        : assetDirectory + "\\test1.mp4";
    const int maxFrames = argc > 3 ? std::max(0, std::stoi(argv[3])) : 0;
    const std::string configFilePath = argc > 4 ? argv[4] : "config.json";

    ANSCustomTL uniformChecker;
    CustomParams customParams;
    if (!uniformChecker.ConfigureParameters(customParams)) {
        std::cerr << "ConfigureParameters failed." << std::endl;
        return 1;
    }

    const std::vector<std::string> expectedNames{
        "ChefHat", "FaceMask", "KitchenGloves", "Apron",
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
    if (customParams.Parameters[4].DefaultValue != "5" ||
        customParams.Parameters[5].DefaultValue != "60" ||
        customParams.Parameters[6].DefaultValue != "5") {
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

    std::string labelMap;
    if (!uniformChecker.Initialize(assetDirectory, 0.25f, labelMap)) {
        std::cerr << "Failed to initialize uniform and pose ONNX models from "
            << assetDirectory << std::endl;
        return 1;
    }
    std::cout << "Event label map: " << labelMap << std::endl;

    cv::VideoCapture capture(videoFilePath);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open test video: " << videoFilePath << std::endl;
        return 1;
    }

    const int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    const int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    cv::namedWindow("FullScreen", cv::WINDOW_NORMAL);
    cv::setWindowProperty("FullScreen", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    cv::Mat frame;
    int frameIndex = 0;
    while (capture.read(frame)) {
        const std::vector<CustomObject> events = uniformChecker.RunInference(frame, "KitchenUniformTest");
        for (const auto& event : events) {
            std::cout << "Frame " << frameIndex << ": " << event.className
                << " track=" << event.trackId << " confidence=" << event.confidence << std::endl;
            cv::rectangle(frame, event.box, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, event.className + " ID " + std::to_string(event.trackId),
                cv::Point(event.box.x, std::max(20, event.box.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        cv::Mat display;
        cv::resize(frame, display, cv::Size(screenWidth, screenHeight));
        cv::imshow("FullScreen", display);
        ++frameIndex;
        if (cv::waitKey(30) == 27 || (maxFrames > 0 && frameIndex >= maxFrames)) break;
    }

    capture.release();
    cv::destroyAllWindows();
    uniformChecker.Destroy();
    std::cout << "Finished after " << frameIndex << " frames." << std::endl;
    return 0;
}
