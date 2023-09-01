#include <cstdio>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

std::unique_ptr<tflite::FlatBufferModel> whisper_model;
std::unique_ptr<tflite::Interpreter> whisper_interpreter;

void test_whisper_tflite() {
    std::string filename = "/Users/yang/CLionProjects/test_tflite/whisper/whisper.tflite";
    whisper_model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    std::cout << 0 << std::endl;
}