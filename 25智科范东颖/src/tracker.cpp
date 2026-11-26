#include "tracker.h"
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

KalmanFilter kf;

void trackerInit(const Point3d& pos)
{

    kf = KalmanFilter(6, 3, 0);

    double dt = 1.0 / 166.0;
    kf.transitionMatrix = (Mat_<float>(6, 6) <<
        1, 0, 0, (float)dt, 0, 0,
        0, 1, 0, 0, (float)dt, 0,
        0, 0, 1, 0, 0, (float)dt,
        0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);

    kf.measurementMatrix = Mat::zeros(3, 6, CV_32F);
    kf.measurementMatrix.at<float>(0, 0) = 1;
    kf.measurementMatrix.at<float>(1, 1) = 1;
    kf.measurementMatrix.at<float>(2, 2) = 1;

    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    kf.statePost = (Mat_<float>(6, 1) <<
        (float)pos.x, (float)pos.y, (float)pos.z, 0, 0, 0);
}

Point3d trackerPredict()
{
    Mat pred = kf.predict();
    return Point3d(pred.at<float>(0), pred.at<float>(1), pred.at<float>(2));
}

void trackerUpdate(const Point3d& meas)
{
    Mat m = (Mat_<float>(3, 1) << (float)meas.x, (float)meas.y, (float)meas.z);
    kf.correct(m);
}

Point3d trackerVelocity()
{

    return Point3d(kf.statePost.at<float>(3),
                   kf.statePost.at<float>(4),
                   kf.statePost.at<float>(5));
}
