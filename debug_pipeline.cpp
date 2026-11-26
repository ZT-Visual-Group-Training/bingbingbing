// ============================================================
// debug_pipeline.cpp —— 检测流程可视化工具（独立程序，不影响 armor_vision）
//
// 它把 detectLights 的四步流程画成一张图，帮你看懂"余光是怎么被挡掉的"：
//   Step1 原图  -> Step2 颜色掩膜(余光也被圈进来)
//   Step3 形状筛选(余光被淘汰) -> Step4 最终灯条
// 下方每张判定卡列出一个轮廓的 6 项参数，红字=没通过的关卡。
// 判定逻辑严格复制 ArmorDetector.cpp 的 adjustRec + detectLights。
//
// CLion 里选 target = debug_pipeline，直接点运行即可（用下面的默认参数）。
// 也可以带参数运行：
//   debug_pipeline [视频路径] [帧号] [输出图片路径]
// 例：
//   debug_pipeline C:/carrie/video.avi 150 D:/demo5/pipeline_truth.png
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

struct Verdict {
    int id;
    cv::Rect bb;
    float area, aspect, angle, crown, height;
    int pts;
    bool okArea, okPts, okAngle, okAspect, okCrown, okHeight;
    bool pass;
};

// 严格复制 ArmorDetector.cpp 的 adjustRec
static cv::RotatedRect adjustRec(cv::RotatedRect r) {
    if (r.size.width > r.size.height) { std::swap(r.size.width, r.size.height); r.angle += 90.f; }
    while (r.angle >= 90.f)  r.angle -= 180.f;
    while (r.angle <  -90.f) r.angle += 180.f;
    return r;
}

int main(int argc, char** argv) {
    std::string video = argc > 1 ? argv[1] : "C:/carrie/video.avi";
    int target = argc > 2 ? std::atoi(argv[2]) : 150;
    std::string outPath = argc > 3 ? argv[3] : "pipeline_truth.png";
    cv::VideoCapture cap(video);
    if (!cap.isOpened()) { std::cerr << "打不开视频\n"; return 1; }
    cv::Mat f;
    for (int i = 0; i <= target; ++i) { if (!cap.read(f) || f.empty()) return 1; }

    cv::Rect roi(450, 690, 290, 150);
    cv::Mat crop = f(roi).clone();

    // === 项目真实参数 ===
    const float lightAreaMin=30.f, lightAngleMax=40.f, aspectRatioMax=0.5f;
    const float crownMin=0.35f, lightHeightMin=20.f; const int minPoints=5;

    cv::Mat hsv; cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
    cv::Mat m1,m2,mask;
    cv::inRange(hsv, cv::Scalar(0,100,80),   cv::Scalar(30,255,255), m1);
    cv::inRange(hsv, cv::Scalar(170,100,80), cv::Scalar(180,255,255), m2);
    cv::bitwise_or(m1,m2,mask);
    cv::Mat k=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::morphologyEx(mask,mask,cv::MORPH_OPEN,k);
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,k,cv::Point(-1,-1),2);

    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(mask,cs,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    std::sort(cs.begin(),cs.end(),[](const std::vector<cv::Point>&a,const std::vector<cv::Point>&b){
        return cv::boundingRect(a).x < cv::boundingRect(b).x; });

    std::vector<Verdict> vs;
    std::vector<cv::RotatedRect> recs;
    for (size_t i=0;i<cs.size();++i){
        Verdict v; v.id=(int)i;
        v.area=(float)cv::contourArea(cs[i]); v.pts=(int)cs[i].size();
        v.bb=cv::boundingRect(cs[i]);
        cv::RotatedRect r=adjustRec(cv::fitEllipse(cs[i])); recs.push_back(r);
        v.aspect=r.size.width/r.size.height;
        v.crown=v.area/r.size.area();
        v.angle=r.angle; v.height=r.size.height;
        v.okArea=v.area>=lightAreaMin; v.okPts=v.pts>minPoints;
        v.okAngle=std::fabs(v.angle)<=lightAngleMax;
        v.okAspect=v.aspect<=aspectRatioMax; v.okCrown=v.crown>=crownMin;
        v.okHeight=v.height>=lightHeightMin;
        v.pass=v.okArea&&v.okPts&&v.okAngle&&v.okAspect&&v.okCrown&&v.okHeight;
        vs.push_back(v);
    }

    std::cout<<"掩膜轮廓 "<<cs.size()<<" 个, 通过 "<<[&](){int n=0;for(auto&v:vs)if(v.pass)n++;return n;}()<<" 根\n";
    for(auto&v:vs) std::cout<<" #"<<v.id<<" bbox("<<v.bb.x<<","<<v.bb.y<<","<<v.bb.width<<"x"<<v.bb.height<<")"
        <<" aspect="<<v.aspect<<" angle="<<v.angle<<" height="<<(int)v.height<<" => "<<(v.pass?"PASS":"REJECT")<<"\n";

    int cw=435,ch=250,th=34;
    auto up=[&](const cv::Mat&m,int chn){cv::Mat b=(chn==1)?cv::Mat():m.clone();
        if(chn==1)cv::cvtColor(m,b,cv::COLOR_GRAY2BGR);cv::Mat big;
        cv::resize(b,big,cv::Size(cw,ch),0,0,cv::INTER_NEAREST);return big;};

    // 列0: 原图
    cv::Mat col0=up(crop,3);
    // 列1: 颜色掩膜 + 轮廓编号（通过=绿, 淘汰=红）
    cv::Mat maskBgr=up(mask,1);
    for(auto&v:vs){
        cv::Scalar c=v.pass?cv::Scalar(0,200,0):cv::Scalar(0,0,220);
        cv::rectangle(maskBgr,cv::Rect(v.bb.x*1.5,v.bb.y*1.67,v.bb.width*1.5,v.bb.height*1.67),c,2);
        cv::putText(maskBgr,"#"+std::to_string(v.id),cv::Point(v.bb.x*1.5,v.bb.y*1.67-4),
            cv::FONT_HERSHEY_SIMPLEX,0.7,c,2);
    }
    // 列2: 形状筛选后掩膜（淘汰的涂黑，通过的留白）
    cv::Mat filtered=cv::Mat::zeros(mask.size(),mask.type());
    for(size_t i=0;i<cs.size();++i) if(vs[i].pass) cv::drawContours(filtered,cs,(int)i,cv::Scalar(255),cv::FILLED);
    cv::Mat col2=up(filtered,1);
    for(auto&v:vs) if(v.pass)
        cv::putText(col2,"#"+std::to_string(v.id)+" OK",cv::Point(v.bb.x*1.5,v.bb.y*1.67-4),
            cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,200,0),2);
    // 列3: 最终灯条拟合（在原图上画绿框）
    cv::Mat col3=crop.clone();
    for(size_t i=0;i<recs.size();++i) if(vs[i].pass){
        cv::Point2f p[4]; recs[i].points(p);
        for(int q=0;q<4;++q) cv::line(col3,p[q],p[(q+1)%4],cv::Scalar(0,255,0),2);
    }
    cv::Mat col3u=up(col3,3);

    cv::Mat canvas(2*(ch+th),cw*4,CV_8UC3,cv::Scalar(245,245,245));
    auto put=[&](int row,int col,const cv::Mat&img,const std::string&title,cv::Scalar tc){
        int x=col*cw,y=row*(ch+th);
        cv::rectangle(canvas,cv::Rect(x,y,cw,th),cv::Scalar(225,225,225),cv::FILLED);
        cv::putText(canvas,title,cv::Point(x+8,y+23),cv::FONT_HERSHEY_SIMPLEX,0.6,tc,1);
        img.copyTo(canvas(cv::Rect(x,y+th,cw,ch)));};
    cv::Scalar dark(40,40,40),red(0,0,200),green(0,120,0),blue(200,100,0);
    put(0,0,col0,"Step1 Original crop",dark);
    put(0,1,maskBgr,"Step2 Color mask (glow #3 IN)",red);
    put(0,2,col2,"Step3 Shape filter (glow OUT)",green);
    put(0,3,col3u,"Step4 Final light bars",green);

    // 行2: 4个轮廓的判定卡
    for(size_t i=0;i<vs.size()&&i<4;++i){
        const Verdict&v=vs[i];
        cv::Mat card=cv::Mat(ch,cw,CV_8UC3,cv::Scalar(255,255,255));
        cv::Scalar col=v.pass?green:red;
        cv::putText(card,"Contour #"+std::to_string(v.id)+(v.pass?"  PASS":"  REJECT"),
            cv::Point(12,28),cv::FONT_HERSHEY_SIMPLEX,0.65,col,2);
        std::string bbox="bbox "+std::to_string(v.bb.width)+"x"+std::to_string(v.bb.height)
            +"  area "+std::to_string((int)v.area);
        cv::putText(card,bbox,cv::Point(12,58),cv::FONT_HERSHEY_SIMPLEX,0.55,dark,1);
        std::string p1="aspect "+std::to_string(v.aspect)+" (max 0.5)";
        cv::putText(card,p1,cv::Point(12,90),cv::FONT_HERSHEY_SIMPLEX,0.55,v.okAspect?green:red,1);
        std::string p2="angle  "+std::to_string(v.angle)+" (max 40)";
        cv::putText(card,p2,cv::Point(12,118),cv::FONT_HERSHEY_SIMPLEX,0.55,v.okAngle?green:red,1);
        std::string p3="height "+std::to_string((int)v.height)+" (min 20)";
        cv::putText(card,p3,cv::Point(12,146),cv::FONT_HERSHEY_SIMPLEX,0.55,v.okHeight?green:red,1);
        std::string p4="crown  "+std::to_string(v.crown)+" (min 0.35)";
        cv::putText(card,p4,cv::Point(12,174),cv::FONT_HERSHEY_SIMPLEX,0.55,v.okCrown?green:red,1);
        std::string tag;
        if(v.pass) tag="=> REAL LIGHT BAR";
        else{
            tag="=> REJECT: ";
            if(!v.okAspect)tag+="aspect ";
            if(!v.okAngle)tag+="angle ";
            if(!v.okHeight)tag+="height";
        }
        cv::putText(card,tag,cv::Point(12,214),cv::FONT_HERSHEY_SIMPLEX,0.55,col,2);
        // 判断是不是余光：横躺(角度超阈值)的淘汰轮廓才算，碎片不标
        if(!v.pass && std::fabs(v.angle)>lightAngleMax){
            cv::rectangle(card,cv::Rect(6,6,cw-12,ch-12),cv::Scalar(0,0,255),3);
            cv::putText(card,"<-- GLOW (horizontal reflection)",cv::Point(12,238),
                cv::FONT_HERSHEY_SIMPLEX,0.45,red,1);
        }
        int x=(int)i*cw,y=ch+th;
        cv::putText(canvas,"#"+std::to_string(v.id)+(v.pass?" PASS":" REJECT"),cv::Point(x+8,y+23),
            cv::FONT_HERSHEY_SIMPLEX,0.6,col,1);
        card.copyTo(canvas(cv::Rect(x,y+th,cw,ch)));
    }

    cv::imwrite(outPath,canvas);
    std::cout<<"[debug_pipeline] 结果已保存: "<<outPath
             <<" ("<<canvas.cols<<"x"<<canvas.rows<<")\n";
    return 0;
}
