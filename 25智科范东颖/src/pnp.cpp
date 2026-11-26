#include "pnp.h"
#include <opencv2/calib3d.hpp>
#include <vector>
using namespace cv;
using namespace std;

static Mat gK;
static Mat gD;
static float gW = 0.135f;
static float gH = 0.056f;

void setPoseParams(const Mat& cameraMatrix, const Mat& distCoeffs,
    float width_mm, float height_mm)
{
    gK = cameraMatrix.clone();
    gD = distCoeffs.clone();
    if (width_mm > 1.0f)  gW = width_mm / 1000.0f;
    if (height_mm > 1.0f) gH = height_mm / 1000.0f;
}

bool getPose(const Point2f pts[4], Mat& tvec, Mat& rvec)
{
    if (gK.empty() || gD.empty())
    {
        return false;
    }

    const float hw = gW / 2.0f;
    const float hh = gH / 2.0f;
    vector<Point3f> objPoints;
    objPoints.emplace_back(-hw, -hh, 0);
    objPoints.emplace_back(hw, -hh, 0);
    objPoints.emplace_back(hw, hh, 0);
    objPoints.emplace_back(-hw, hh, 0);

    vector<Point2f> imgPoints(pts, pts + 4);

    bool pnpRet = solvePnP(
        objPoints,
        imgPoints,
        gK,
        gD,
        rvec,
        tvec,
        false,
        SOLVEPNP_ITERATIVE
    );
    if (!pnpRet)
    {
        return false;
    }

    vector<Point2f> reprojPoints;
    projectPoints(objPoints, rvec, tvec, gK, gD, reprojPoints);

    float totalErr = 0.0f;
    for (int i = 0; i < 4; i++)
    {
        totalErr += (float)norm(reprojPoints[i] - imgPoints[i]);
    }
    float avgErr = totalErr / 4.0f;

    if (avgErr > 15.0f)
    {
        return false;
    }

    return true;
}
