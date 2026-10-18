// Plantify: run the exported ONNX model with OpenCV C++ (no Python / PyTorch needed).
// Same steps as src/infer_opencv.py.
//
//   ./plantify_infer ../exports/b2_imagenet_full.onnx leaf1.jpg [leaf2.jpg ...]

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model.onnx> <image> [<image> ...]\n";
        return 1;
    }
    std::string modelPath = argv[1];
    std::string folder = modelPath.substr(0, modelPath.find_last_of('/') + 1);

    // Class names, one per line, in the model's output order.
    std::vector<std::string> labels;
    std::ifstream labelFile(folder + "labels.txt");
    for (std::string line; std::getline(labelFile, line);)
        if (!line.empty()) labels.push_back(line);

    // Input size from model_info.json (cv::FileStorage can read JSON).
    cv::FileStorage info(folder + "model_info.json", cv::FileStorage::READ);
    std::vector<int> inputShape;
    info["input"] >> inputShape;  // [1, 3, H, W]
    int size = inputShape.empty() ? 128 : inputShape[3];

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    std::cout << "Loaded " << modelPath << " (" << labels.size() << " classes, input " << size << "x" << size << ")\n";

    for (int i = 2; i < argc; ++i) {
        cv::Mat bgr = cv::imread(argv[i]);
        if (bgr.empty()) {
            std::cout << "\nCould not read " << argv[i] << "\n";
            continue;
        }

        // Same preprocessing as training: RGB, resize, scale to [0,1], ImageNet mean/std.
        cv::Mat rgb, resized, x;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(size, size), 0, 0, cv::INTER_AREA);
        resized.convertTo(x, CV_32FC3, 1.0 / 255.0);
        cv::subtract(x, cv::Scalar(0.485, 0.456, 0.406), x);
        cv::divide(x, cv::Scalar(0.229, 0.224, 0.225), x);
        cv::Mat blob = cv::dnn::blobFromImage(x);  // HxWx3 -> 1x3xHxW

        int64 t0 = cv::getTickCount();
        net.setInput(blob);
        cv::Mat logits = net.forward();  // 1 x num_classes
        double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();

        // Softmax turns logits into probabilities.
        std::vector<float> probs(logits.begin<float>(), logits.end<float>());
        float maxLogit = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (float& p : probs) {
            p = std::exp(p - maxLogit);
            sum += p;
        }
        for (float& p : probs) p /= sum;

        // Indices sorted by probability, highest first.
        std::vector<int> order(probs.size());
        for (size_t k = 0; k < order.size(); ++k) order[k] = static_cast<int>(k);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[a] > probs[b]; });

        std::cout << "\n" << argv[i] << "\n";
        for (int k = 0; k < 3 && k < static_cast<int>(order.size()); ++k)
            std::printf("  %-30s %.4f\n", labels[order[k]].c_str(), probs[order[k]]);
        std::printf("  forward pass: %.1f ms\n", ms);
    }
    return 0;
}
