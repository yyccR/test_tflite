#include <cstdio>
#include <iostream>
#include <fstream>
#include <thread>
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_N_MEL       80
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_MEL_LEN     3000

std::unique_ptr<tflite::FlatBufferModel> whisper_model;
std::unique_ptr<tflite::Interpreter> whisper_interpreter;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int golden_generated_ids[21] = {50257,50362,1770,13,2264,346,353,318,262,46329,286,262,3504,6097,11,290,356,389,9675,284,7062};

struct whisper_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 51864;


    std::map<id, token> id_to_token;

    id token_eot  = 50256;
    id token_sot  = 50257;
    id token_prev = 50360;
    id token_solm = 50361; // ??
    id token_not  = 50362; // no timestamps
    id token_beg  = 50363;

    // available tasks
    static const id token_translwordate  = 50358;
    static const id token_transcribe = 50359;

    bool is_multilingual() const {
        return n_vocab == 51865;
    }

};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct whisper_mel {
    int n_len;
    int n_mel;

    std::vector<float> data;
};

// 定义whisper词汇表,滤波器,mel
whisper_vocab g_vocab;
whisper_filters filters;
whisper_mel mel;

// Discrete Fourier Transform, 离散傅里叶变换
void dft(const std::vector<float> & in, std::vector<float> & out) {
    int N = in.size();

    out.resize(N*2);

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            float angle = 2*M_PI*k*n/N;
            re += in[n]*cos(angle);
            im -= in[n]*sin(angle);
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
void fft(const std::vector<float> & in, std::vector<float> & out) {
    out.resize(in.size()*2);

    int N = in.size();

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N%2 == 1) {
        dft(in, out);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(in[i]);
        } else {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft);
    fft(odd, odd_fft);

    for (int k = 0; k < N/2; k++) {
        float theta = 2*M_PI*k/N;

        float re = cos(theta);
        float im = -sin(theta);

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + N/2) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + N/2) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L92-L124
bool log_mel_spectrogram(
        const float * samples,
        const int n_samples,
        const int sample_rate,
        const int fft_size,
        const int fft_step,
        const int n_mel,
        const int n_threads,
        const whisper_filters & filters,
        whisper_mel & mel) {

    // Hanning window
    std::vector<float> hann;
    hann.resize(fft_size);
    for (int i = 0; i < fft_size; i++) {
        hann[i] = 0.5*(1.0 - cos((2.0*M_PI*i)/(fft_size)));
    }

    mel.n_mel = n_mel;
    mel.n_len = (n_samples)/fft_step;
    mel.data.resize(mel.n_mel*mel.n_len);

    const int n_fft = 1 + fft_size/2;

    //printf("%s: n_samples = %d, n_len = %d\n", __func__, n_samples, mel.n_len);
    //printf("%s: recording length: %f s\n", __func__, (float) n_samples/sample_rate);

    std::vector<std::thread> workers(n_threads);
    for (int iw = 0; iw < n_threads; ++iw) {
        workers[iw] = std::thread([&](int ith) {
            std::vector<float> fft_in;
            fft_in.resize(fft_size);
            for (int i = 0; i < fft_size; i++) {
                fft_in[i] = 0.0;
            }

            std::vector<float> fft_out;
            fft_out.resize(2*fft_size);

            for (int i = ith; i < mel.n_len; i += n_threads) {
                const int offset = i*fft_step;

                // apply Hanning window
                for (int j = 0; j < fft_size; j++) {
                    if (offset + j < n_samples) {
                        fft_in[j] = hann[j]*samples[offset + j];
                    } else {
                        fft_in[j] = 0.0;
                    }
                }

                // FFT -> mag^2
                fft(fft_in, fft_out);

                for (int j = 0; j < fft_size; j++) {
                    fft_out[j] = (fft_out[2*j + 0]*fft_out[2*j + 0] + fft_out[2*j + 1]*fft_out[2*j + 1]);
                }
                for (int j = 1; j < fft_size/2; j++) {
                    //if (i == 0) {
                    //    printf("%d: %f %f\n", j, fft_out[j], fft_out[fft_size - j]);
                    //}
                    fft_out[j] += fft_out[fft_size - j];
                }
                if (i == 0) {
                    //for (int j = 0; j < fft_size; j++) {
                    //    printf("%d: %e\n", j, fft_out[j]);
                    //}
                }

                // mel spectrogram
                for (int j = 0; j < mel.n_mel; j++) {
                    double sum = 0.0;

                    for (int k = 0; k < n_fft; k++) {
                        sum += fft_out[k]*filters.data[j*n_fft + k];
                    }
                    if (sum < 1e-10) {
                        sum = 1e-10;
                    }

                    sum = log10(sum);

                    mel.data[j*mel.n_len + i] = sum;
                }
            }
        }, iw);
    }

    for (int iw = 0; iw < n_threads; ++iw) {
        workers[iw].join();
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }
    //printf("%s: max = %f\n", __func__, mmax);

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    return true;
}

//const char * whisper_token_to_str(int token) {
//    return g_vocab.id_to_token.at(token).c_str();
//}



void test_whisper_tflite() {
    std::string model_file = "/Users/yang/CLionProjects/test_tflite/whisper/whisper.tflite";
    std::string vocab_file = "/Users/yang/CLionProjects/test_tflite/whisper/filters_vocab_gen.bin";
    std::string wav_file = "/Users/yang/CLionProjects/test_tflite/data/audio/test.wav";

    // 加载词汇表
    std::ifstream file(vocab_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << vocab_file << std::endl;
        return;
    } else {
        std::cout << "read vocab/mel_filters file: " << vocab_file << std::endl;
    }
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    // 检查魔数
    if (magic != 0x5553454e) {
        std::cerr << "Invalid vocab file (bad magic)" << std::endl;
        return;
    }

    // 读取 mel filters 数据
    file.read(reinterpret_cast<char*>(&filters.n_mel), sizeof(filters.n_mel));
    file.read(reinterpret_cast<char*>(&filters.n_fft), sizeof(filters.n_fft));
    filters.data.resize(filters.n_mel * filters.n_fft);
    file.read(reinterpret_cast<char*>(filters.data.data()), filters.data.size() * sizeof(float));

    // 读取词汇表
    int32_t n_vocab = 0;
    std::string word;

    file.read(reinterpret_cast<char*>(&n_vocab), sizeof(n_vocab));
    g_vocab.n_vocab = n_vocab;

    for (int i = 0; i < n_vocab; i++) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0');
        file.read(&word[0], len);
//        std::cout << i << ": " << word << std::endl;
        g_vocab.id_to_token[i] = word;
    }

    // 进行其他操作，如添加额外的词汇
    g_vocab.n_vocab = 51864;
    if (g_vocab.is_multilingual()) {
        g_vocab.token_eot++;
        g_vocab.token_sot++;
        g_vocab.token_prev++;
        g_vocab.token_solm++;
        g_vocab.token_not++;
        g_vocab.token_beg++;
    }
    for (int i = n_vocab; i < g_vocab.n_vocab; i++) {
        if (i > g_vocab.token_beg) {
            word = "[_TT_" + std::to_string(i - g_vocab.token_beg) + "]";
        } else if (i == g_vocab.token_eot) {
            word = "[_EOT_]";
        } else if (i == g_vocab.token_sot) {
            word = "[_SOT_]";
        } else if (i == g_vocab.token_prev) {
            word = "[_PREV_]";
        } else if (i == g_vocab.token_not) {
            word = "[_NOT_]";
        } else if (i == g_vocab.token_beg) {
            word = "[_BEG_]";
        } else {
            word = "[_extra_token_" + std::to_string(i) + "]";
        }
        g_vocab.id_to_token[i] = word;
        // printf("%s: g_vocab[%d] = '%s'\n", __func__, i, word.c_str());
    }

    // 关闭vocab文件
    file.close();

    // 加载模型
    whisper_model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*whisper_model, resolver)(&whisper_interpreter);
    TFLITE_MINIMAL_CHECK(whisper_interpreter != nullptr);
    TFLITE_MINIMAL_CHECK(whisper_interpreter->AllocateTensors() == kTfLiteOk);
    std::cout << "read whisper model: " << model_file << std::endl;

    // 加载音频
    std::vector<float> pcmf32;
    drwav wav;
    size_t audio_dataSize=0;
    char* audio_buffer = nullptr;
    drwav_init_file(&wav, wav_file.c_str(), NULL);
    std::cout << "read wav file: " << wav_file << std::endl;
    std::cout << "wav file channels: " << wav.channels << ", sample rate: " << wav.sampleRate << ", totalPCMFrameCount: " << wav.totalPCMFrameCount << std::endl;
    int n = wav.totalPCMFrameCount;
    std::vector<int16_t> pcm16;
    pcm16.resize(n*wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);
    // convert to mono, float
    pcmf32.resize(n);
    if (wav.channels == 1) {
        for (int i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[i])/32768.0f;
//            std::cout << "pcmf32[i]" << pcmf32[i] << std::endl;
        }
    } else {
        for (int i = 0; i < n; i++) {
            pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
        }
    }

    //Hack if the audio file size is less than 30ms append with 0's
    pcmf32.resize((WHISPER_SAMPLE_RATE*WHISPER_CHUNK_SIZE),0);
    const auto processor_count = std::thread::hardware_concurrency();
    log_mel_spectrogram(pcmf32.data(), pcmf32.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, processor_count,filters, mel);


    // 模型输入
    float* input = whisper_interpreter->typed_input_tensor<float>(whisper_interpreter->inputs()[0]);
    memcpy(input, mel.data.data(), mel.n_mel*mel.n_len*sizeof(float));
    // 模型计算
    TFLITE_MINIMAL_CHECK(whisper_interpreter->Invoke() == kTfLiteOk);
    // 模型输出
    for(auto &i : whisper_interpreter->outputs()){
        std::cout << i << std::endl;
    }
    TfLiteTensor* output_tensor = whisper_interpreter->tensor(whisper_interpreter->outputs()[0]);
    int last_dimension = output_tensor->dims->data[output_tensor->dims->size - 1];
    int* output_data = whisper_interpreter->typed_output_tensor<int>(0);
    std::string text = "";
    std::string word_add;
    for (int i = 0; i < last_dimension; i++) {
        //printf("%d\t",output_int[i]);
        if(output_data[i] == g_vocab.token_eot){
            break;
        }
        if((output_data[i] !=50257)&& (output_data[i] !=50362))
            text += g_vocab.id_to_token.at(output_data[i]).c_str();
//            text += whisper_token_to_str(output_data[i]);
    }

    std::cout << "translate result: " << text << std::endl;
}