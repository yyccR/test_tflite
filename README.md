## TFLite c++ in PC

### mac构建tflite c++
1. 构建 `libtensorflowlite.dylib`
```
bazel build -c opt  --config=macos --cpu=darwin //tensorflow/lite:libtensorflowlite.dylib
bazel build -c opt //tensorflow/lite:libtensorflowlite.dylib
```

2. 构建 `libtensorflowlite_gpu_delegate.dylib`, 需要先构建1步骤

(1) `uname -a`查看cpu架构, 如果是`x86_64`, 替换`tensorflow/bazel-tensorflow/external/cpuinfo/BUILD.bazel`里面 `"cpu": "darwin"` 为`"cpu": "darwin_x86_64"`
(2) 在根目录编译,需指定`--cxxopt=-std=c++17`
```
bazel build -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++17 //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib --apple_platform_type=macos --cpu=darwin_x86_64 --macos_cpus=x86_64

install_name_tool -id "/Users/yang/CLionProjects/test_tflite/tflite-2.10.0/tflite2.10.0_lib/mac-os/gpu/tensorflow_lite_gpu_dylib_bin" tensorflow_lite_gpu_dylib.dylib
```