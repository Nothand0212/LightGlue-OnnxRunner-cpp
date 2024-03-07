# ONNX Runtime Light Glue Example

[ **简体中文**  /  [English](doc/README-en.md)]

一个简单的ONNX Runtime Light Glue示例。

测试环境：

- Ubuntu 20.04
- ONNX Runtime onnxruntime-linux-x64-gpu-1.16.3 
- ONNX Runtime onnxruntime-linux-x64-1.16.3
- CUDA 11.7
- SPDLOG 1.13.0
- OpenCV 4.9.0

![结果](doc/Matches_screenshot_20.01.2024.png)

## 编译运行

### 编译

```sh
git clone https://github.com/Nothand0212/LightGlue-OnnxRunner-cpp.git
cd LightGlue-OnnxRunner-cpp
mkdir build && cd build
cmake.. && make -j
```

### 运行

- 修改`config/param.json`里的路径，包括matcher_path、extractor_path、combiner_path、image_src_path和image_dst_path等。

## 更新日志

- 2024-03-07
  - 1. 将spdlog改为纯头文件版本，解决编译问题
  - 2. 添加onnxruntime-linux-x64-gpu-1.16.3到thirdparty
  - 3. 修改了CMakeLists.txt，目的是想做到尽量开箱即用。当然目前CUDA环境和OpenCV环境还是需要手动安装的。
- 2024-02-02
  - 1. 增加了单独的`Matcher`类
- 2024-01-28
  - 1. 增加单独的`Extractor`类
- 2024-01-24
  - 1. 增加非端到端模型的支持（只支持super point 和 light glue）
- 2024-01-20
  - 1. 目前只支持super point 和 light glue 的端到端模型

## 遗留BUG

- 2024-01-28
  - 1. 2024-01-24的遗留的bug，应该是模型的问题。在`decoupled_onnx_test`中，如果用模型`superpoint_lightglue_fused_fp16.onnx`则不会出现相同的问题。
- 2024-01-24
  - 1. 在GPU环境下运行时，某些图片(如项目里data/left/21.png)会报错`Non-zero status code returned while running MultiHeadAttention node. Name:'MultiHeadAttention_1' Status Message: packed QKV format is not implemented for current GPU. Please disable it in fusion options.`。但是在CPU环境下运行时，不会报错。

## 参考

- [LightGlue](https://github.com/cvg/LightGlue)
- [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
- [LightGlue-OnnxRunner](https://github.com/OroChippw/LightGlue-OnnxRunner)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [SPDLOG](https://github.com/gabime/spdlog)