# ANSCustomUsingPhone

**ANSCustomUsingPhone** is an extension module (custom model) for the ANS Engine designed to detect phone usage behaviors. The project is built as a DLL and compressed into a zip format. Once imported into ANSVIS, the custom model can be used for real-time video inference. It utilizes a multi-model approach combining:
* YOLO Object Detection (for phone detection)
* YOLO Pose Estimation (for human keypoint extraction)
* AI Gesture Logic (analyzing head orientation and wrist distance)


*Naming convention: File DLL has to start with ANSCustom{ProjectName}.dll. For example ANSCustomUsingPhone.dll to work with ANSVIS*

## 🖥 Requirements
* Windows 10/11
* Visual Studio 2022  
* C++17  
* OpenCV ≥ 4.1  
* CUDA 12.x (recommended)  
* ANSLib
- Link to downnload ANSLib: [ANSLibs](https://storage.googleapis.com/anscenter-public-resources/ANSCENTER_LIB/ANSLibs.zip)
- Location:
![image](https://github.com/user-attachments/assets/50900c4d-e560-4d3a-9d3a-776a65bf4ccc)
* ANS Runtime Engine
- Link to download: [ANS Runtime Engine](https://storage.googleapis.com/anscenter-public-resources/runtime-engine/Version1.6.0/ANS_Run-Time_Engine.zip)

## 🔧 Build Instructions
1. Clone project:
  ```bash
  git clone https://github.com/ANSCENTER-AIDev/ANSUsingPhone.git
  ```
2. Open file on Visual Studio
  ```
  ANSUsingPhone.sln
  ```
3. Configuration
### Configure build
  ```
  Release | x64
  ```
### Include Directories
![image](https://github.com/user-attachments/assets/f7683a2f-d745-495e-b5e2-cff348aec416)


### Library Directories
![image](https://github.com/user-attachments/assets/d07ebb0b-4050-4d2e-a582-a72b90b548f4)

### Linker
![image](https://github.com/user-attachments/assets/c9244a4c-46c5-4267-ae77-9f6891b6ed62)
4. Build solution (Ctrl + Shift + B)
  ```
  ./x64/Release/ANSCustomUsingPhone.dll
  ```

## 🚀 Usage (Example Traffic Light Project)
### Build the custom model
![image](https://github.com/user-attachments/assets/e84725ef-9668-43a2-8845-61f0f32c3ecc)

### Prepare for the model 
![image](https://github.com/user-attachments/assets/6961cfa8-b93e-4971-aa46-ae1417dfe6c6)

*Note: Zip file with password: **AnsDemoModels20@!***
![image](https://github.com/user-attachments/assets/24f319b1-a002-44c1-8f87-14e93e1671db)
## 📁 Project Structure
```
ANSCustomUsingPhone/
├── ANSUsingPhone/
|    ├── ANSUsingPhone-UnitTest/
|    |   └── ANSUsingPhone-UnitTest.cpp
|    ├── ANSUsingPhone.sln
|    └── Dll/
|        ├── ANSUsingPhone.cpp
|        ├── ANSUsingPhone.h
|        ├── ANSByteTrack.cpp
|        └── ANSByteTrack.h
├── ByteTrack
├── Eigen3
├── nlohmann
├── .gitignore
└── README.md
```
