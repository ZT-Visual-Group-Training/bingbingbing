// =============================================================================
// aegis_node.cpp —— Aegis Vision 的 ROS2 包装（可选，不参与考核默认构建）
//
// 订阅：/image_raw                  sensor_msgs/Image
// 发布：/aegis/target               geometry_msgs/PointStamped（相机系 XYZ，米）
//       /aegis/detections           std_msgs/Float32MultiArray
//                                   [track_id, display_id, number, x, y, z,
//                                    distance, reproj_error, confidence]
//       /aegis/debug_image          sensor_msgs/Image（带 HUD 的调试图，可选）
//
// 用法（ROS2 Humble）：
//   colcon build --packages-select aegis_ros
//   ros2 launch aegis_ros aegis.launch.py
// =============================================================================
#include <algorithm>
#include <memory>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "aegis/hud.hpp"
#include "aegis/pipeline.hpp"

class AegisNode : public rclcpp::Node {
 public:
  AegisNode() : Node("aegis_vision") {
    const std::string configPath =
        declare_parameter<std::string>("config", "config/aegis.yaml");
    const std::string imageTopic =
        declare_parameter<std::string>("image_topic", "/image_raw");
    publishDebug_ = declare_parameter<bool>("publish_debug_image", false);

    cfg_ = aegis::loadConfig(configPath);
    pipeline_ = std::make_unique<aegis::Pipeline>(cfg_);
    hud_ = std::make_unique<aegis::Hud>(cfg_);

    targetPub_ = create_publisher<geometry_msgs::msg::PointStamped>("/aegis/target", 10);
    detPub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/aegis/detections", 10);
    if (publishDebug_) {
      debugPub_ = create_publisher<sensor_msgs::msg::Image>("/aegis/debug_image", 2);
    }

    sub_ = create_subscription<sensor_msgs::msg::Image>(
        imageTopic, rclcpp::SensorDataQoS(),
        std::bind(&AegisNode::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Aegis Vision ready: %s -> /aegis/target",
                imageTopic.c_str());
  }

 private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImagePtr cvPtr;
    try {
      cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    const double stampMs = static_cast<double>(msg->header.stamp.sec) * 1000.0 +
                           static_cast<double>(msg->header.stamp.nanosec) / 1e6;
    aegis::FrameResult result = pipeline_->process(cvPtr->image, ++frame_, stampMs);

    const aegis::Track* primary = nullptr;
    for (const auto& t : result.tracks) {
      if (t.id == result.primaryTrack) primary = &t;
    }
    if (primary == nullptr || !primary->pose.valid) return;

    geometry_msgs::msg::PointStamped p;
    p.header = msg->header;
    p.point.x = primary->pose.tvec[0];
    p.point.y = primary->pose.tvec[1];
    p.point.z = primary->pose.tvec[2];
    targetPub_->publish(p);

    std_msgs::msg::Float32MultiArray det;
    det.data = {static_cast<float>(primary->id),
                static_cast<float>(primary->displayId),
                static_cast<float>(primary->number),
                static_cast<float>(primary->pose.tvec[0]),
                static_cast<float>(primary->pose.tvec[1]),
                static_cast<float>(primary->pose.tvec[2]),
                static_cast<float>(primary->pose.distance),
                static_cast<float>(primary->pose.reprojError),
                primary->score};
    detPub_->publish(det);

    if (publishDebug_ && debugPub_) {
      cv::Mat vis = cvPtr->image.clone();
      hud_->draw(vis, result, hudState_);
      ++hudState_.frameIndex;
      hudState_.fps = 1000.0 / std::max(0.5, result.timing.total);
      hudState_.fpsHistory.push_back(static_cast<float>(hudState_.fps));
      if (hudState_.fpsHistory.size() > 120) hudState_.fpsHistory.pop_front();
      auto out = cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg();
      debugPub_->publish(*out);
    }
  }

  aegis::AppConfig cfg_;
  std::unique_ptr<aegis::Pipeline> pipeline_;
  std::unique_ptr<aegis::Hud> hud_;
  aegis::HudState hudState_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr targetPub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr detPub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugPub_;
  bool publishDebug_ = false;
  int frame_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AegisNode>());
  rclcpp::shutdown();
  return 0;
}

