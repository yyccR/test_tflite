
#include <cstdio>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
//#include "tensorflow/lite/delegates/gpu/delegate.h"
//#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"
//#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
//#include "tensorflow/lite/delegates/gpu/metal_delegate_internal.h"
////#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

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

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}

void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}

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

//    TfLiteDelegate* delegate = TFLGpuDelegateCreate(nullptr);
//    std::cout << interpreter->ModifyGraphWithDelegate(delegate) << std::endl;

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
//    float* input = interpreter->typed_input_tensor<float>(interpreter->inputs()[0]);
    float* input = interpreter->typed_input_tensor<float>(interpreter->inputs()[0]);
//    std::string imfile = "/Users/yang/CLionProjects/test_tflite/data/images/img.jpg";
    std::string imfile = "/Users/yang/Downloads/JPEGImages/Cats_Test0.jpg";
    cv::Mat im = cv::imread(imfile, cv::IMREAD_COLOR);
    cv::Mat im2;
    cv::resize(im, im2, cv::Size(320, 320));
    im2.convertTo(im2, CV_32F, 1.0/255);
    float* imPointer = im2.ptr<float>(0);

//    interpreter->typed_input_tensor<float>(0) = imPointer;
    std::cout << im2.rows << " " << im2.cols << " " << im2.channels() << std::endl;
    memcpy(input, imPointer, im2.rows*im2.cols*im2.channels()*sizeof(int32_t));

    TfLiteIntArray* f = interpreter->input_tensor(0)->dims;
    for(int i =0; i<f->size;i++){
        std::cout << "input[" <<i<<"]" << f->data[i] << std::endl;
    }


    float w_scale = (float)im.cols / (float)320;
    float h_scale = (float)im.rows / (float)320;
//    img.convertTo(img, CV_32F, 255.f/input_std);
//    cv::subtract(img, cv::Scalar(input_mean/input_std), img);
//    float* in = img.ptr<float>(0);
//    memcpy(out, in, img.rows * img.cols * sizeof(float));

//    cv::imshow("", im2);
//    cv::waitKey(0);
//    input = (float*)im2.data;
//    memcpy(tflite::GetTensorData<float>(input), im2.data,sizeof(float) * im2.size[1] * im2.size[2] * im2.size[3]);

//    std::cout << " " <<interpreter->input_tensor(0)->type << std::endl;
//    for(auto i : interpreter->outputs()){
//        std::cout << i << std::endl;
//    }


//    // Run inference
    TfLiteStatus status = interpreter->Invoke();
    std::cout << "tflite status: " << status << std::endl;
//    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//    printf("\n\n=== Post-invoke Interpreter State ===\n");
//    tflite::PrintInterpreterState(interpreter.get());

    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

//    TfLiteIntArray* ff = interpreter->output_tensor(0)->dims;
//    for(int i =0; i<ff->size;i++){
//        std::cout << ff->data[i] << std::endl;
//    }

    float* output = interpreter->typed_output_tensor<float>(0);
    std::cout << "output: " << output << std::endl;
    std::vector<BoxInfo> boxes;
    for(int i = 0; i < 6300; i++){

        if(*(output+i*85+4) > 0.2){
            int cur_label = 0;
            float score = *(output+i*85+4+1);
            for (int label = 0; label < 80; label++)
            {
                //LOGD("decode_infer label %d",label);
                //LOGD("decode_infer score %f",scores[label]);
                if (*(output+i*85+5+label) > score)
                {
                    score = *(output+i*85+5+label);
                    cur_label = label;
                }
            }

            float x = *(output+i*85+0)* 320.0f * w_scale;
            float y = *(output+i*85+1)* 320.0f * h_scale;
            float w = *(output+i*85+2)* 320.0f * w_scale;
            float h = *(output+i*85+3)* 320.0f * h_scale;

            boxes.push_back(BoxInfo{
                    (float)std::max(0.0, x-w/2.0),
                    (float)std::max(0.0, y-h/2.0),
                    (float)std::min((float)im.cols, (float)(x+w/2.0)),
                    (float)std::min((float)im.rows, (float)(y+h/2.0)),
                    *(output+i*85+4),
                    cur_label
            });
//            std::cout << " x1: " << (float)std::max(0.0, x-w/2.0) <<
//            " y1: " << (float)std::max(0.0, y-h/2.0) <<
//            " x2: " << (float)std::min(320.0, x+w/2.0) <<
//            " y2: " << (float)std::min(320.0, y+h/2.0) <<
//            " socre: " << *(output+i*85+4) <<
//            " label: " << cur_label << std::endl;
        }
    }

    nms(boxes, 0.6);
    for(auto &box: boxes){
        std::cout << " x1: " << box.x1 <<
                  " y1: " << box.y1 <<
                  " x2: " << box.x2 <<
                  " y2: " << box.y2 <<
                  " socre: " << box.score <<
                  " label: " << box.label << std::endl;
    }
    draw_coco_bboxes(im, boxes);
    cv::waitKey(0);

//    TfLiteIntArray* ff = interpreter->output_tensor(0)->dims;
//    for(int i =0; i<ff->size;i++){
//        std::cout << "output["<<i<<"]" << ff->data[i] << std::endl;
//    }
//
//    float a[2][2] = {{1,2},{3,4}};
//    float (*b)[2] = a;
//    float *c = b[0];
//    for(int i = 0;i<4;i++){
//        std::cout << *(c+i) << std::endl;
//    }

    return 0;
}