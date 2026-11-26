from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config = DeclareLaunchArgument(
        "config", default_value="config/aegis.yaml", description="Path to aegis.yaml"
    )
    image_topic = DeclareLaunchArgument(
        "image_topic", default_value="/image_raw", description="Camera image topic"
    )

    node = Node(
        package="aegis_ros",
        executable="aegis_node",
        name="aegis_vision",
        output="screen",
        parameters=[
            {"config": LaunchConfiguration("config")},
            {"image_topic": LaunchConfiguration("image_topic")},
            {"publish_debug_image": True},
        ],
    )

    return LaunchDescription([config, image_topic, node])
