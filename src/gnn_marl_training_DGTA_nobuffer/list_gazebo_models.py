#!/usr/bin/env python3
"""
诊断脚本：列出当前Gazebo中的所有模型
用于检查是否有残留的静态障碍物
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetWorldProperties, GetModelList
import sys

def main():
    rclpy.init()
    node = Node('list_models')

    # 创建服务客户端
    client = node.create_client(GetModelList, '/get_model_list')

    if not client.wait_for_service(timeout_sec=5.0):
        print("❌ Gazebo服务 /get_model_list 不可用（Gazebo可能未启动）")
        node.destroy_node()
        rclpy.shutdown()
        return

    # 请求模型列表
    request = GetModelList.Request()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    if future.result() is not None:
        response = future.result()
        models = response.model_names

        print(f"\n📋 Gazebo中当前有 {len(models)} 个模型:\n")

        # 分类统计
        static_boxes = []
        static_obs = []
        dynamic_obs = []
        robots = []
        others = []

        for name in models:
            if name.startswith('static_box_'):
                static_boxes.append(name)
            elif name.startswith('static_obs_'):
                static_obs.append(name)
            elif name.startswith('dyn_obs_'):
                dynamic_obs.append(name)
            elif name.startswith('tb3_'):
                robots.append(name)
            else:
                others.append(name)

        if static_boxes:
            print(f"🟫 棕色方块 (static_box_*): {len(static_boxes)} 个")
            for name in sorted(static_boxes):
                print(f"   - {name}")
            print()

        if static_obs:
            print(f"⚠️  旧命名静态障碍物 (static_obs_*): {len(static_obs)} 个")
            for name in sorted(static_obs):
                print(f"   - {name}")
            print()

        if dynamic_obs:
            print(f"🔴 动态障碍物 (dyn_obs_*): {len(dynamic_obs)} 个")
            for name in sorted(dynamic_obs):
                print(f"   - {name}")
            print()

        if robots:
            print(f"🤖 机器人 (tb3_*): {len(robots)} 个")
            for name in sorted(robots):
                print(f"   - {name}")
            print()

        if others:
            print(f"📦 其他模型: {len(others)} 个")
            for name in sorted(others):
                print(f"   - {name}")
            print()

        # 警告信息
        if static_obs:
            print("⚠️  检测到旧命名的static_obs_*障碍物！")
            print("   这些可能是之前残留的，会导致A*规划器无法感知")
            print("   已修复：reset时会同时删除static_obs_*和static_box_*\n")

        if len(static_boxes) + len(static_obs) > 8:
            print("⚠️  静态障碍物数量超过8个，可能有残留！\n")

    else:
        print("❌ 服务调用失败")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
