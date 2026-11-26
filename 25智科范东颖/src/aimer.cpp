#include "aimer.h"
#include <cmath>

using namespace cv;

namespace {

constexpr double kPi      = 3.14159265358979323846;
constexpr double kRad2Deg = 180.0 / kPi;

double gBulletSpeed = 20.0;
double gLatency     = 0.03;
double gGravity     = 9.8;

Mat gR;
Mat gT;

bool hasExtrinsics()
{
    return !gR.empty() && gR.rows == 3 && gR.cols == 3;
}

Point3d cameraToGimbal(const Point3d& p)
{
    if (!hasExtrinsics())
        return p;
    Mat pc = (Mat_<double>(3, 1) << p.x, p.y, p.z);
    Mat pg = gR * pc;
    if (!gT.empty() && gT.total() == 3)
        pg += gT;
    return Point3d(pg.at<double>(0), pg.at<double>(1), pg.at<double>(2));
}

void anglesFromCamera(const Point3d& p, double& yawDeg, double& pitchDeg)
{
    yawDeg   = std::atan2(p.x, p.z) * kRad2Deg;
    pitchDeg = -std::atan2(p.y, p.z) * kRad2Deg;
}

void anglesFromGimbal(const Point3d& p, double& yawDeg, double& pitchDeg)
{
    const double horiz = std::hypot(p.x, p.y);
    yawDeg   = std::atan2(-p.y, p.x) * kRad2Deg;
    pitchDeg = std::atan2(p.z, horiz) * kRad2Deg;
}

}

void setAimParams(double bulletSpeed, double latencySec, double gravity)
{
    if (bulletSpeed > 1.0)  gBulletSpeed = bulletSpeed;
    if (latencySec >= 0.0)  gLatency = latencySec;
    if (gravity > 0.0)      gGravity = gravity;
}

void setAimExtrinsics(const Mat& R_camera2gimbal, const Mat& t_camera2gimbal)
{
    gR = R_camera2gimbal.clone();
    gT = t_camera2gimbal.clone();
}

AimResult solveAim(const Point3d& pos, const Point3d& vel, bool hasVelocity)
{
    AimResult res;

    if (pos.z <= 0.3)
        return res;
    double dist = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
    if (dist < 0.5 || dist > 20.0)
        return res;

    Point3d aimPos = pos;
    double tFlight = dist / gBulletSpeed;
    if (hasVelocity)
    {
        for (int i = 0; i < 3; ++i)
        {
            const double leadTime = gLatency + tFlight;
            aimPos = Point3d(pos.x + vel.x * leadTime,
                             pos.y + vel.y * leadTime,
                             pos.z + vel.z * leadTime);
            if (aimPos.z <= 0.3)
                return res;

            dist = std::sqrt(aimPos.x * aimPos.x + aimPos.y * aimPos.y +
                             aimPos.z * aimPos.z);
            const double pitchRad = -std::atan2(aimPos.y, aimPos.z);
            double cosPitch = std::cos(pitchRad);
            if (cosPitch < 0.2) cosPitch = 0.2;
            tFlight = dist / (gBulletSpeed * cosPitch);
        }
    }

    const double drop = 0.5 * gGravity * tFlight * tFlight;
    double dropDeg = 0.0;
    if (dist > 0.5)
        dropDeg = std::atan2(drop, dist) * kRad2Deg;

    double yawCam = 0.0, pitchCam = 0.0;
    anglesFromCamera(aimPos, yawCam, pitchCam);
    pitchCam += dropDeg;

    double yawG = 0.0, pitchG = 0.0;
    anglesFromGimbal(cameraToGimbal(aimPos), yawG, pitchG);
    pitchG += dropDeg;

    res.ok          = true;
    res.yawCam      = yawCam;
    res.pitchCam    = pitchCam;
    res.yawGimbal   = yawG;
    res.pitchGimbal = pitchG;
    res.predPos     = aimPos;
    res.tFlight     = tFlight;
    res.leadTime    = gLatency + tFlight;
    return res;
}
