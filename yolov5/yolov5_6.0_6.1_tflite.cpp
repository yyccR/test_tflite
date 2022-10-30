
#include <cstdio>
#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

//#define TFLITE_MINIMAL_CHECK(x)                              \
//  if (!(x)) {                                                \
//    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
//    exit(1);                                                 \
//  }

std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;

int main() {

    std::string filename = "/Users/yang/CLionProjects/test_tflite/models/yolov5s-coco-320.tflite";
//    const char[] filename = '/Users/yang/CLionProjects/test_tflite/models/yolov5s-coco-320.tflite';

    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());
//    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if(true){
        TfLiteDelegate* delegate = TfLiteGpuDelegateV2Create(nullptr);
        std::cout << interpreter->ModifyGraphWithDelegate(delegate) << std::endl;
    }

//    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
//    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
//    printf("=== Pre-invoke Interpreter State ===\n");
//    tflite::PrintInterpreterState(interpreter.get());
    interpreter->AllocateTensors();

    // Fill input buffers
    // TODO(user): Insert code to fill input tensors.
    // Note: The buffer of the input tensor with index `i` of type T can
    // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
    float* input = interpreter->typed_input_tensor<float>(interpreter->inputs()[0]);
    std::cout << interpreter->tensor(0) << std::endl;


    // Run inference
//    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//    printf("\n\n=== Post-invoke Interpreter State ===\n");
//    tflite::PrintInterpreterState(interpreter.get());

    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

    return 0;
}