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

<!-- ![结果](doc/Matches_screenshot_20.01.2024.png) -->
<div align="center">
  <img src="doc/superpoint-lightglue.gif" width="800">
</div>
>

## 编译运行

### 编译

- 需要修改`CMakeLists.txt`文件，指定ONNX Runtime的路径。可以在[这里下载](https://github.com/microsoft/onnxruntime/releases)，选择1.16.3版本。
- 下载后解压缩，将文件夹的路径修改为`ONNXRUNTIME_ROOTDIR`变量的值。

```cmake
set(ONNXRUNTIME_ROOTDIR /home/lin/Projects/onnxruntime-linux-x64-gpu-1.16.3)
```

```sh
git clone https://github.com/Nothand0212/LightGlue-OnnxRunner-cpp.git
cd LightGlue-OnnxRunner-cpp
mkdir build && cd build
cmake.. && make -j
```

### 运行

- 修改`config/param.json`里的路径为自己的路径，包括
- 1. matcher_path
- 2. extractor_path
- 3. combiner_path
- 4. image_src_path
- 5. image_dst_path
- 6. 其他

## 更新日志

- 2024-03-07
  - 1. 修改`Extractor`的提取特征点的处理逻辑，提取后先不恢复到原图的坐标，否则会导致匹配结果不准确。
  - 2. 修改运行逻辑，运行时需要指定配置文件路径，例如： `./matcher_test /home/lin/Projects/LightGlue-OnnxRunner-cpp/config/param.json`
  - 3. 将spdlog改为纯头文件版本，解决编译问题
- 2024-02-02
  - 1. 增加了单独的`Matcher`类
- 2024-01-28
  - 1. 增加单独的`Extractor`类
- 2024-01-24
  - 1. 增加非端到端模型的支持（只支持super point 和 light glue）
- 2024-01-20
  - 1. 目前只支持super point 和 light glue 的端到端模型

~~## 遗留问题~~

~~- 2024-01-28~~
~~  - 1. 2024-01-24的遗留的bug，应该是模型的问题。在`decoupled_onnx_test`中，如果用模型`superpoint_lightglue_fused_fp16.onnx`则不会出现相同的问题。~~
~~- 2024-01-24~~
~~  - 1. 在GPU环境下运行时，某些图片(如项目里data/left/21.png)会报错`Non-zero status code returned while running MultiHeadAttention node. Name:'MultiHeadAttention_1' Status Message: packed QKV format is not implemented for current GPU. Please disable it in fusion options.`。但是在CPU环境下运行时，不会报错。~~

## 参考

- [LightGlue](https://github.com/cvg/LightGlue)
- [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
- [LightGlue-OnnxRunner](https://github.com/OroChippw/LightGlue-OnnxRunner)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [SPDLOG](https://github.com/gabime/spdlog)
