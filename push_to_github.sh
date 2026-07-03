#!/bin/bash
# 推送脚本 - 用于将本地代码推送到GitHub
# 使用方法: ./push_to_github.sh

set -e

echo "=========================================="
echo "准备推送代码到GitHub"
echo "=========================================="

cd "/home/wj/work/multi-robot-exploration-rl copy git"

# 显示当前状态
echo ""
echo "📊 当前状态:"
git log --oneline -3
echo ""

# 配置git参数以支持大仓库
echo "⚙️  配置Git参数..."
git config http.postBuffer 524288000  # 500MB buffer
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999

echo ""
echo "🚀 开始推送..."
echo "注意: 由于仓库较大(~950MB)，推送可能需要5-10分钟"
echo ""

# 尝试推送
if git push origin main; then
    echo ""
    echo "=========================================="
    echo "✅ 推送成功！"
    echo "=========================================="
    echo ""
    echo "GitHub仓库: https://github.com/Bluestone-work/TII"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 推送失败"
    echo "=========================================="
    echo ""
    echo "可能的原因:"
    echo "1. 网络连接不稳定"
    echo "2. 仓库太大导致超时"
    echo ""
    echo "建议解决方案:"
    echo "1. 换个网络环境（比如校园网/家庭网络）"
    echo "2. 使用GitHub Desktop图形界面工具"
    echo "3. 晚上网络较好时再试"
    echo ""
    exit 1
fi
