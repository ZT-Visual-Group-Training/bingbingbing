#include "detector.h"
#include "pnp.h"
#include "tracker.h"
#include "number_classifier.h"
#include "utils.h"
#include "aimer.h"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
using namespace cv;
using namespace std;

static void usage(const char* exe)
{
    cout << "Usage:\n"
         << "  " << exe << " <video> [out_video] [out_csv] [options]\n\n"
         << "  out_video : default output/result.mp4\n"
         << "  out_csv   : default output/result.csv\n\n"
         << "Options:\n"
         << "  --config-dir DIR : directory with camera/armor/aim yaml (default: config)\n"
         << "  --max-frames N   : process at most N frames (default: all)\n"
         << "  --no-gui         : do not open a display window (batch/benchmark mode)\n\n"
         << "Example:\n"
         << "  " << exe << " data/video.avi output/result_1.mp4 output/result_1.csv --no-gui\n";
}

static bool openWriter(VideoWriter& w, const string& path, const Size& sz, double fps, string& usedPath)
{
    string ext = filesystem::path(path).extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);

    vector<pair<string, int> > cand;
    if (ext == ".mp4")
        cand.push_back(make_pair(path, VideoWriter::fourcc('m', 'p', '4', 'v')));
    else if (ext == ".avi")
        cand.push_back(make_pair(path, VideoWriter::fourcc('M', 'J', 'P', 'G')));
    else
    {
        cand.push_back(make_pair(path + ".mp4", VideoWriter::fourcc('m', 'p', '4', 'v')));
        cand.push_back(make_pair(path + ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G')));
    }

    for (size_t i = 0; i < cand.size(); ++i)
    {
        const string& p = cand[i].first;
        int cc = cand[i].second;
        if (w.open(p, CAP_FFMPEG, cc, fps, sz)) { usedPath = p; return true; }
        if (w.open(p, 0, cc, fps, sz))         { usedPath = p; return true; }
    }
    return false;
}

int main(int argc, char** argv)
{
    vector<string> pos;
    bool noGui = false;
    string configDir = "config";
    int maxFrames = 0;

    for (int i = 1; i < argc; ++i)
    {
        string a = argv[i];
        if (a == "--no-gui" || a == "--nogui") noGui = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--config-dir" && i + 1 < argc) { configDir = argv[++i]; }
        else if (a == "--max-frames" && i + 1 < argc) { maxFrames = atoi(argv[++i]); }
        else if (!a.empty() && a[0] == '-') { cerr << "[ERROR] unknown option: " << a << endl; usage(argv[0]); return 1; }
        else pos.push_back(a);
    }

    if (pos.empty())
    {
        cerr << "[WARN] no video argument, using default demo path" << endl;
        pos.push_back("2026-11-13_18-25-25_462305/video/video.avi");
    }

    const string videoPath = pos[0];
    string outVideo = (pos.size() > 1) ? pos[1] : "output/result.mp4";
    string outCsv   = (pos.size() > 2) ? pos[2] : "output/result.csv";

    CameraParams cam;
    ArmorParams armorParams;
    if (!loadCameraParams(configDir + "/camera.yaml", cam))
    {
        cerr << "[ERROR] failed to load " << configDir << "/camera.yaml" << endl;
        return 5;
    }
    if (!loadArmorParams(configDir + "/armor.yaml", armorParams))
    {
        cerr << "[ERROR] failed to load " << configDir << "/armor.yaml" << endl;
        return 5;
    }
    AimParams aimParams;
    if (!loadAimParams(configDir + "/aim.yaml", aimParams))
        cerr << "[WARN] config/aim.yaml missing, using default aim parameters" << endl;

    setPoseParams(cam.camera_matrix, cam.distort_coeffs,
                  armorParams.armor_width_mm, armorParams.armor_height_mm);
    setEnemyColor(armorParams.enemy_color);
    setAimParams(aimParams.bullet_speed, aimParams.latency_ms / 1000.0, aimParams.gravity);
    setAimExtrinsics(cam.R_camera2gimbal, cam.t_camera2gimbal);

    cout << "[INFO] config dir  : " << configDir << endl;
    cout << "[INFO] camera      : " << cam.image_width << "x" << cam.image_height
         << "  fx=" << cam.camera_matrix.at<double>(0, 0)
         << "  fy=" << cam.camera_matrix.at<double>(1, 1) << endl;
    cout << "[INFO] armor size  : " << armorParams.armor_width_mm << " x "
         << armorParams.armor_height_mm << " mm" << endl;
    cout << "[INFO] enemy color : " << armorParams.enemy_color << endl;
    cout << "[INFO] aim params  : bullet=" << aimParams.bullet_speed << " m/s, latency="
         << aimParams.latency_ms << " ms, g=" << aimParams.gravity << endl;

    VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cerr << "[ERROR] cannot open video: " << videoPath << endl;
        return 2;
    }

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 1.0 || fps > 1000.0)
    {
        cerr << "[WARN] invalid fps from video (" << fps << "), fallback to 30" << endl;
        fps = 30.0;
    }
    int totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);

    Mat img;
    if (!cap.read(img) || img.empty())
    {
        cerr << "[ERROR] cannot read first frame" << endl;
        return 2;
    }
    const Size frameSize = img.size();

    if (cam.image_width > 0 && cam.image_height > 0 &&
        (frameSize.width != cam.image_width || frameSize.height != cam.image_height))
    {
        cerr << "[ERROR] frame size " << frameSize.width << "x" << frameSize.height
             << " != camera intrinsics " << cam.image_width << "x" << cam.image_height
             << ", PnP distance would be wrong" << endl;
        return 6;
    }

    cout << "[INFO] video       : " << videoPath << endl;
    cout << "[INFO] fps         : " << fps << endl;
    cout << "[INFO] total frames: " << totalFrames << endl;
    cout << "[INFO] frame size  : " << frameSize.width << "x" << frameSize.height << endl;

    if (!outVideo.empty())
    {
        filesystem::path p(outVideo);
        if (p.has_parent_path()) filesystem::create_directories(p.parent_path());
    }
    if (!outCsv.empty())
    {
        filesystem::path p(outCsv);
        if (p.has_parent_path()) filesystem::create_directories(p.parent_path());
    }

    VideoWriter writer;
    string usedVideoPath = outVideo;
    if (!outVideo.empty() && !openWriter(writer, outVideo, frameSize, fps, usedVideoPath))
    {
        cerr << "[ERROR] cannot open VideoWriter for " << outVideo << endl;
        return 3;
    }
    if (!outVideo.empty())
        cout << "[INFO] output video : " << usedVideoPath << " (fps=" << fps
             << ", " << frameSize.width << "x" << frameSize.height << ")" << endl;

    ofstream csv;
    if (!outCsv.empty())
    {
        csv.open(outCsv.c_str());
        if (!csv.is_open())
        {
            cerr << "[WARN] cannot open csv file: " << outCsv << endl;
        }
        else
        {
            csv << "frame,time_s,detected,num_armors,number,is_primary,x,y,z,kf_x,kf_y,kf_z,"
                   "yaw_cam,pitch_cam,yaw_gimbal,pitch_gimbal,pred_x,pred_y,pred_z,t_flight,lead_time,ms\n";
            cout << "[INFO] output csv   : " << outCsv << endl;
        }
    }

    if (!loadDigitModel()) cout << "[WARN] digit model load failed" << endl;
    if (!loadYolo())       cout << "[WARN] YOLO model load failed" << endl;

    Mat mask;
    bool tracked = false;
    int kfFrames = 0;
    int frame = 0;
    double msSum = 0;
    int expectedFrames = (maxFrames > 0 && maxFrames < totalFrames) ? maxFrames : totalFrames;

    while (!img.empty())
    {
        if (frame >= expectedFrames) break;

        int64 t0 = getTickCount();

        findColor(img, mask);
        vector<Armor> armors;
        getArmors(img, mask, armors);
        if (armors.empty()) detectByYolo(img, armors);

        int primary = -1;
        float bestDist = 1e9f;
        Point2f imgCenter(img.cols / 2.0f, img.rows / 2.0f);
        for (int i = 0; i < (int)armors.size(); i++)
        {
            float d = (float)norm(armors[i].center - imgCenter);
            if (d < bestDist) { bestDist = d; primary = i; }
        }

        vector<Point3d> poses(armors.size());
        vector<char> poseOk(armors.size(), 0);
        Point3d kfPos(0, 0, 0);
        bool kfOk = false;
        AimResult aim;
        bool aimOk = false;

        for (int i = 0; i < (int)armors.size(); i++)
        {
            Armor& ar = armors[i];

            for (int k = 0; k < 4; k++)
            {
                circle(img, ar.pts[k], 6, Scalar(0, 255, 0), -1);
                line(img, ar.pts[k], ar.pts[(k + 1) % 4], Scalar(0, 255, 0), 2);
            }
            circle(img, ar.center, 6, Scalar(0, 0, 255), -1);
            putText(img, ar.number, Point((int)ar.center.x - 10, (int)ar.center.y - 20),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

            Mat tvec, rvec;
            if (getPose(ar.pts, tvec, rvec))
            {
                Point3d pos(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                poses[i] = pos;
                poseOk[i] = 1;

                if (i == primary)
                {
                    if (!tracked) { trackerInit(pos); tracked = true; kfFrames = 0; }
                    kfPos = trackerPredict();
                    trackerUpdate(pos);
                    kfFrames++;
                    Point3d vel = trackerVelocity();
                    kfOk = true;

                    char buf[128];
                    snprintf(buf, sizeof(buf), "raw xyz: %.3f %.3f %.3f m", pos.x, pos.y, pos.z);
                    putText(img, buf, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
                    snprintf(buf, sizeof(buf), "kf  xyz: %.3f %.3f %.3f m", kfPos.x, kfPos.y, kfPos.z);
                    putText(img, buf, Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                    snprintf(buf, sizeof(buf), "vel: %.2f %.2f %.2f m/s", vel.x, vel.y, vel.z);
                    putText(img, buf, Point(10, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);

                    aim = solveAim(pos, vel, kfFrames >= 3);
                    aimOk = aim.ok;
                }
                else
                {
                    char buf[128];
                    snprintf(buf, sizeof(buf), "%.3f %.3f %.3f", pos.x, pos.y, pos.z);
                    putText(img, buf, Point((int)ar.center.x, (int)ar.center.y + 25),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
                }
            }
        }

        if (armors.empty())
        {
            tracked = false;
            kfFrames = 0;
            putText(img, "no armor", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        if (aimOk)
        {
            char abuf[192];
            snprintf(abuf, sizeof(abuf), "yaw: %.2f deg   pitch: %.2f deg", aim.yawCam, aim.pitchCam);
            putText(img, abuf, Point(10, 140), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 2);
            snprintf(abuf, sizeof(abuf), "yawG: %.2f  pitchG: %.2f  t_f: %.3f s",
                     aim.yawGimbal, aim.pitchGimbal, aim.tFlight);
            putText(img, abuf, Point(10, 165), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 0, 255), 1);

            if (aim.predPos.z > 0.1)
            {
                int pu = (int)(cam.camera_matrix.at<double>(0, 0) * aim.predPos.x / aim.predPos.z
                               + cam.camera_matrix.at<double>(0, 2));
                int pv = (int)(cam.camera_matrix.at<double>(1, 1) * aim.predPos.y / aim.predPos.z
                               + cam.camera_matrix.at<double>(1, 2));
                circle(img, Point(pu, pv), 5, Scalar(255, 0, 0), -1);
            }

            const double yawRad   = aim.yawCam   * CV_PI / 180.0;
            const double pitchRad = aim.pitchCam * CV_PI / 180.0;
            int cu = (int)(cam.camera_matrix.at<double>(0, 0) * std::tan(yawRad)
                           + cam.camera_matrix.at<double>(0, 2));
            int cvp = (int)(cam.camera_matrix.at<double>(1, 2)
                            - cam.camera_matrix.at<double>(1, 1) * std::tan(pitchRad));
            line(img, Point(cu - 14, cvp), Point(cu + 14, cvp), Scalar(255, 0, 255), 2);
            line(img, Point(cu, cvp - 14), Point(cu, cvp + 14), Scalar(255, 0, 255), 2);
        }

        double ms = (getTickCount() - t0) * 1000.0 / getTickFrequency();
        char tbuf[64];
        snprintf(tbuf, sizeof(tbuf), "%.1f ms (%.0f fps)", ms, ms > 0 ? 1000.0 / ms : 0.0);
        putText(img, tbuf, Point(10, 105), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        char nbuf[32];
        snprintf(nbuf, sizeof(nbuf), "armors: %d", (int)armors.size());
        putText(img, nbuf, Point(10, 125), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        if (!noGui)
        {
            imshow("img", img);
            int key = waitKey(1);
            if (key == 'q' || key == 27) break;
        }

        if (writer.isOpened()) writer.write(img);

        if (csv.is_open())
        {
            int isPrimary = 0;
            string num;
            double px = 0, py = 0, pz = 0, kx = 0, ky = 0, kz = 0;
            for (int i = 0; i < (int)armors.size(); i++)
            {
                if (!poseOk[i]) continue;
                if (i == primary)
                {
                    num = armors[i].number;
                    isPrimary = 1;
                    px = poses[i].x; py = poses[i].y; pz = poses[i].z;
                    if (kfOk) { kx = kfPos.x; ky = kfPos.y; kz = kfPos.z; }
                    break;
                }
            }

            std::ostringstream row;
            row << frame << ',' << fixed << setprecision(6) << (double)frame / fps << ','
                << (isPrimary ? 1 : 0) << ',' << (int)armors.size() << ','
                << num << ',' << (isPrimary ? 1 : 0) << ','
                << fixed << setprecision(4) << px << ',' << py << ',' << pz << ',';
            if (isPrimary && kfOk)
                row << fixed << setprecision(4) << kx << ',' << ky << ',' << kz << ',';
            else
                row << ",,,";
            if (isPrimary && aimOk)
            {
                row << fixed << setprecision(3)
                    << aim.yawCam << ',' << aim.pitchCam << ','
                    << aim.yawGimbal << ',' << aim.pitchGimbal << ','
                    << aim.predPos.x << ',' << aim.predPos.y << ',' << aim.predPos.z << ','
                    << aim.tFlight << ',' << aim.leadTime << ',';
            }
            else
            {
                row << ",,,,,,,,,";
            }
            row << fixed << setprecision(3) << ms;
            csv << row.str() << '\n';
        }

        msSum += ms;
        frame++;
        if (frame % 300 == 0)
            cout << "[INFO] processed " << frame << " / " << expectedFrames << " frames" << endl;

        if (!cap.read(img)) break;
    }

    if (writer.isOpened()) writer.release();
    if (csv.is_open()) csv.close();
    cap.release();
    if (!noGui) destroyAllWindows();

    double avgMs = frame > 0 ? msSum / frame : 0.0;
    cout << "[DONE] processed " << frame << " / " << expectedFrames << " frames, avg "
         << fixed << setprecision(2) << avgMs << " ms/frame ("
         << (avgMs > 0 ? 1000.0 / avgMs : 0) << " fps)" << endl;
    if (!outVideo.empty()) cout << "[DONE] video: " << usedVideoPath << endl;
    if (!outCsv.empty())   cout << "[DONE] csv  : " << outCsv << endl;

    if (expectedFrames > 0 && frame != expectedFrames)
    {
        cerr << "[ERROR] processed frames (" << frame << ") != expected (" << expectedFrames
             << "), output video is INCOMPLETE" << endl;
        return 4;
    }
    cout << "[OK] complete run" << endl;
    return 0;
}
