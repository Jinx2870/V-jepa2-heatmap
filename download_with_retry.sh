#!/bin/zsh

# goexo 自动重试下载脚本
# 命令中断（失败）后自动再次运行，成功则结束
# 按 Ctrl+C 可以停止

echo "开始下载，如果中断会自动重试"
echo "按 Ctrl+C 可以停止"
echo ""

retry_count=0

while true; do
    retry_count=$((retry_count + 1))
    
    echo "=========================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 第 $retry_count 次运行"
    echo "=========================================="
    
    # 执行命令
    goexo -o . --parts downscaled_takes/448 --views exo -y
    
    # 获取命令的退出码
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 下载成功完成！"
        echo "=========================================="
        exit 0
    else
        echo ""
        echo "=========================================="
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 命令中断（退出码: $exit_code）"
        echo "将在 3 秒后自动重试..."
        echo "=========================================="
        echo ""
        sleep 3
    fi
done
