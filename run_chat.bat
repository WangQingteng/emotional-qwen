@echo off
chcp 65001 >nul
echo ============================================================
echo 情绪价值助手 - 启动脚本
echo ============================================================
echo.

REM 检查模型文件是否存在
if not exist "./output/emotional_qwen/final_model" (
    echo [错误] 未找到训练好的模型
    echo 请先运行 run_train.bat 进行训练
    pause
    exit /b 1
)

echo 启动交互式对话模式...
echo 输入你的问题，输入 quit 退出
echo.

python scripts/inference.py --model_path ./output/emotional_qwen/final_model --mode interactive

pause