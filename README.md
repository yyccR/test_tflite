## TFLite c++ in PC

### mac构建tflite c++
1. 构建 `libtensorflowlite.dylib`
```
bazel build -c opt //tensorflow/lite:libtensorflowlite.dylib
```

2. 构建 `libtensorflowlite_gpu_delegate.dylib`
```
bazel build -c opt //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.dylib
```