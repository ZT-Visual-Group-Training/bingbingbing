#include "number_classifier.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;

string modelPath = "model/mlp.onnx";
string labelPath = "model/label.txt";

dnn::Net net;
vector<string> classNames;
bool modelLoaded = false;

bool loadDigitModel()
{
    net = dnn::readNetFromONNX(modelPath);
    if (net.empty())
    {
        cout << "[ERROR] failed to load digit model: " << modelPath << endl;
        return false;
    }
    ifstream f(labelPath);
    string line;
    while (getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (!line.empty()) classNames.push_back(line);
    }
    modelLoaded = true;
    return true;
}

string classifyNumber(const Mat& src, const Point2f pts[4], double& confidence)
{
    confidence = 0;
    if (!modelLoaded) return "";

    const int warp_w = 32, warp_h = 28, light_len = 12;
    int top_y = (warp_h - light_len) / 2 - 1;
    int bottom_y = top_y + light_len;

    Point2f lights[4] = { pts[3], pts[0], pts[1], pts[2] };
    Point2f target[4] = {
        Point2f(0, bottom_y), Point2f(0, top_y),
        Point2f(warp_w - 1, top_y), Point2f(warp_w - 1, bottom_y) };

    Mat warped;
    Mat M = getPerspectiveTransform(lights, target);
    warpPerspective(src, warped, M, Size(warp_w, warp_h));

    Mat roi = warped(Rect((warp_w - 20) / 2, 0, 20, 28));

    cvtColor(roi, roi, COLOR_BGR2GRAY);
    threshold(roi, roi, 0, 255, THRESH_BINARY | THRESH_OTSU);

    roi.convertTo(roi, CV_32F, 1.0 / 255.0);
    Mat blob = dnn::blobFromImage(roi);

    net.setInput(blob);
    Mat out = net.forward();
    float maxp = *max_element(out.begin<float>(), out.end<float>());
    Mat softmax;
    exp(out - maxp, softmax);
    softmax /= (float)sum(softmax)[0];

    double conf = 0;
    Point maxLoc;
    minMaxLoc(softmax.reshape(1, 1), 0, &conf, 0, &maxLoc);
    int id = maxLoc.x;
    if (id < 0 || id >= (int)classNames.size()) return "";
    confidence = conf;
    return classNames[id];
}
