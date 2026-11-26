#include "detector.h"
#include "number_classifier.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#ifdef HAVE_OPENVINO
#include <openvino/openvino.hpp>
#endif
using namespace cv;
using namespace std;

int hmin = 0, smin = 110, vmin = 180;
int hmax = 35, smax = 255, vmax = 255;
int hmin2 = 170, hmax2 = 180;
int minArea = 60;
int minRatio = 20;
double digitConfThreshold = 0.5;
float maxGap = 4.0f;

dnn::Net yoloNet;
bool yoloLoaded = false;
static bool gYoloUseOV = false;
#ifdef HAVE_OPENVINO
static ov::CompiledModel gYoloOVModel;
static ov::InferRequest gYoloOVRequest;
#endif
float yoloConf = 0.3f;
float yoloNms = 0.45f;
int yoloNc = 14;
const char* armorNames[14] = {
    "B1","B2","B3","B4","B5","BO","BS",
    "R1","R2","R3","R4","R5","RO","RS"
};

static int gEnemyColor = 2;

void setEnemyColor(const std::string& color)
{
    if (color == "blue")      gEnemyColor = 1;
    else if (color == "both") gEnemyColor = 2;
    else                      gEnemyColor = 0;
}

void findColor(const Mat& img, Mat& mask)
{
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    Mat red, blue;
    if (gEnemyColor == 0 || gEnemyColor == 2)
    {
        Mat m2;
        inRange(imgHSV, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), red);
        inRange(imgHSV, Scalar(hmin2, smin, vmin), Scalar(hmax2, smax, vmax), m2);
        red = red | m2;
    }
    if (gEnemyColor == 1 || gEnemyColor == 2)
    {
        inRange(imgHSV, Scalar(100, smin, vmin), Scalar(130, smax, vmax), blue);
    }

    {
        Mat ch[3];
        split(img, ch);
        Mat gb, rb, rScore, bScore, rAuto, bAuto;
        max(ch[1], ch[2], gb);
        max(ch[0], ch[2], rb);
        subtract(ch[2], gb, rScore);
        subtract(ch[0], rb, bScore);
        double tR = threshold(rScore, rAuto, 0, 255, THRESH_BINARY | THRESH_OTSU);
        if (tR < 40) threshold(rScore, rAuto, 40, 255, THRESH_BINARY);
        double tB = threshold(bScore, bAuto, 0, 255, THRESH_BINARY | THRESH_OTSU);
        if (tB < 40) threshold(bScore, bAuto, 40, 255, THRESH_BINARY);
        if (gEnemyColor == 0 || gEnemyColor == 2) red = red | rAuto;
        if (gEnemyColor == 1 || gEnemyColor == 2) blue = blue | bAuto;
    }

    if (gEnemyColor == 2)      mask = red | blue;
    else if (gEnemyColor == 1) mask = blue;
    else                       mask = red;

    Mat kOpen = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kOpen);
    Mat kClose = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kClose);
}

void findLightBars(const Mat& mask, vector<LightBar>& bars)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < minArea) continue;
        RotatedRect r = minAreaRect(contours[i]);
        float len = max(r.size.width, r.size.height);
        float wid = min(r.size.width, r.size.height);
        if (wid < 1 || len / wid < minRatio / 10.0) continue;

        float angle = r.angle;
        if (r.size.width < r.size.height) angle += 90;
        angle = fmodf(angle, 180);
        if (angle < 0) angle += 180;

        Vec4f line;
        fitLine(contours[i], line, DIST_L2, 0, 0.01, 0.01);
        Point2f dir(line[0], line[1]);
        Point2f org(line[2], line[3]);
        float tmin = FLT_MAX, tmax = -FLT_MAX;
        for (int k = 0; k < (int)contours[i].size(); k++)
        {
            Point2f q((float)contours[i][k].x, (float)contours[i][k].y);
            float t = (q - org).dot(dir);
            if (t < tmin) tmin = t;
            if (t > tmax) tmax = t;
        }
        Point2f p1 = org + dir * tmin;
        Point2f p2 = org + dir * tmax;
        if (p1.y > p2.y) swap(p1, p2);

        LightBar b;
        b.center = (p1 + p2) * 0.5f;
        b.p1 = p1;
        b.p2 = p2;
        b.length = (float)norm(p2 - p1);
        b.angle = angle;
        bars.push_back(b);
    }
}

void getArmors(const Mat& img, const Mat& mask, vector<Armor>& armors)
{
    armors.clear();
    vector<LightBar> bars;
    findLightBars(mask, bars);

    for (int i = 0; i < (int)bars.size(); i++)
    {
        for (int j = i + 1; j < (int)bars.size(); j++)
        {
            LightBar& a = bars[i];
            LightBar& b = bars[j];
            LightBar* left = (a.center.x < b.center.x) ? &a : &b;
            LightBar* right = (a.center.x < b.center.x) ? &b : &a;

            float dx = right->center.x - left->center.x;
            if (dx < 1) continue;
            float avgLen = (left->length + right->length) / 2;
            if (avgLen < 1) continue;

            float dAngle = fabs(left->angle - right->angle);
            dAngle = min(dAngle, 180.0f - dAngle);
            if (dAngle > 40) continue;
            if (fabs(left->center.y - right->center.y) > avgLen * 2) continue;
            float lenRatio = max(left->length, right->length) / min(left->length, right->length);
            if (lenRatio > 3.0f) continue;
            float gap = dx / avgLen;
            if (gap < 1.0f || gap > maxGap) continue;

            Armor ar;
            ar.pts[0] = left->p1;
            ar.pts[1] = right->p1;
            ar.pts[2] = right->p2;
            ar.pts[3] = left->p2;
            ar.center = (ar.pts[0] + ar.pts[1] + ar.pts[2] + ar.pts[3]) * 0.25f;

            double conf = 0;
            string num = classifyNumber(img, ar.pts, conf);
            if (num == "negative" || num == "") continue;
            if (conf < digitConfThreshold) continue;
            if (num != "1" && num != "2" && num != "3" && num != "4" && num != "5") continue;

            ar.number = num;
            armors.push_back(ar);
        }
    }
}

static void sortCorners(Point2f pts[4])
{
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            if (pts[i].x > pts[j].x) { Point2f t = pts[i]; pts[i] = pts[j]; pts[j] = t; }
    if (pts[0].y > pts[1].y) { Point2f t = pts[0]; pts[0] = pts[1]; pts[1] = t; }
    if (pts[2].y > pts[3].y) { Point2f t = pts[2]; pts[2] = pts[3]; pts[3] = t; }
    Point2f tl = pts[0], bl = pts[1], tr = pts[2], br = pts[3];
    pts[0] = tl; pts[1] = tr; pts[2] = br; pts[3] = bl;
}

bool loadYolo()
{
#ifdef HAVE_OPENVINO
    try {
        ov::Core core;
        auto model = core.read_model("model/armor_pose416.onnx");
        model->reshape(ov::PartialShape{1, 3, 416, 416});
        gYoloOVModel = core.compile_model(model, "CPU");
        gYoloOVRequest = gYoloOVModel.create_infer_request();
        gYoloUseOV = true;
        yoloLoaded = true;
        std::cout << "[INFO] YOLO backend: OpenVINO" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[WARN] OpenVINO YOLO load failed: " << e.what() << std::endl;
        gYoloUseOV = false;
        yoloLoaded = false;
    }
#endif
    try {
        yoloNet = dnn::readNetFromONNX("model/armor_pose416.onnx");
    } catch (const cv::Exception& e) {
        std::cerr << "YOLO model load failed: " << e.what() << std::endl;
        yoloLoaded = false;
        return false;
    }
    yoloNet.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    yoloNet.setPreferableTarget(dnn::DNN_TARGET_CPU);
    yoloLoaded = !yoloNet.empty();
    std::cout << "[INFO] YOLO backend: OpenCV dnn" << std::endl;
    return yoloLoaded;
}

void detectByYolo(const Mat& img, vector<Armor>& armors)
{
    armors.clear();
    if (!yoloLoaded) return;
    int h0 = img.rows, w0 = img.cols;

    Mat blob = dnn::blobFromImage(img, 1.0 / 255.0, Size(416, 416), Scalar(), true, false);
    Mat out;
#ifdef HAVE_OPENVINO
    if (gYoloUseOV)
    {
        auto input = gYoloOVRequest.get_input_tensor();
        std::memcpy(input.data<float>(), blob.ptr<float>(), 3 * 416 * 416 * sizeof(float));
        gYoloOVRequest.infer();
        auto output = gYoloOVRequest.get_output_tensor(0);
        auto shape = output.get_shape();
        if (shape.size() == 3)
            out = Mat((int)shape[1], (int)shape[2], CV_32F,
                      (void*)output.data<const float>()).clone();
        else if (shape.size() == 2)
            out = Mat((int)shape[0], (int)shape[1], CV_32F,
                      (void*)output.data<const float>()).clone();
    }
#endif
    if (out.empty())
    {
        yoloNet.setInput(blob);
        out = yoloNet.forward();
    }

    Mat o;
    if (out.dims == 3)
        o = Mat(out.size[1], out.size[2], CV_32F, out.data);
    else
        o = out;
    int nb = o.cols;

    float sx = (float)w0 / 416.0f;
    float sy = (float)h0 / 416.0f;

    vector<Rect> boxes;
    vector<float> confs;
    vector<int> classes;
    vector<vector<Point2f> > kptsList;

    for (int i = 0; i < nb; i++)
    {
        float maxScore = 0;
        int maxCls = -1;
        for (int c = 0; c < yoloNc; c++)
        {
            float s = o.at<float>(4 + c, i);
            if (s > maxScore) { maxScore = s; maxCls = c; }
        }
        if (maxScore < yoloConf) continue;

        float cx = o.at<float>(0, i);
        float cy = o.at<float>(1, i);
        float w = o.at<float>(2, i);
        float h = o.at<float>(3, i);
        int x1 = (int)((cx - w / 2) * sx);
        int y1 = (int)((cy - h / 2) * sy);
        int bw = (int)(w * sx);
        int bh = (int)(h * sy);
        boxes.push_back(Rect(x1, y1, bw, bh));
        confs.push_back(maxScore);
        classes.push_back(maxCls);

        vector<Point2f> kp;
        for (int k = 0; k < 4; k++)
        {
            float kx = o.at<float>(4 + yoloNc + k * 2, i) * sx;
            float ky = o.at<float>(4 + yoloNc + k * 2 + 1, i) * sy;
            kp.push_back(Point2f(kx, ky));
        }
        kptsList.push_back(kp);
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, confs, yoloConf, yoloNms, indices);

    for (int ii = 0; ii < (int)indices.size(); ii++)
    {
        int idx = indices[ii];
        Armor ar;
        for (int k = 0; k < 4; k++) ar.pts[k] = kptsList[idx][k];
        sortCorners(ar.pts);
        ar.center = (ar.pts[0] + ar.pts[1] + ar.pts[2] + ar.pts[3]) * 0.25f;
        ar.number = armorNames[classes[idx]];
        armors.push_back(ar);
    }
}
