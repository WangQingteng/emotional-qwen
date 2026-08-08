@echo off
chcp 65001 >nul
echo ============================================================
echo 情绪价值模型微调 - 一键启动脚本
echo ============================================================
echo.

REM 检查 Python 环境
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] 检查 Python 环境...
python --version

echo.
echo [2/5] 安装依赖包...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [警告] 依赖安装可能有警告，继续执行...
)

echo.
echo [3/5] 下载并构建情绪对话数据集...
python scripts/build_dataset.py --datasets emochat emotional_qa --add_custom
if %errorlevel% neq 0 (
    echo [错误] 数据集构建失败
    pause
    exit /b 1
)

echo.
echo [4/5] 预处理数据...
python scripts/preprocess.py --data_path ./data/combined_emotional_sft.jsonl --model_name Qwen/Qwen2.5-0.5B-Instruct
if %errorlevel% neq 0 (
    echo [错误] 数据预处理失败
    pause
    exit /b 1
)

echo.
echo [5/5] 开始 QLoRA 微调训练...
echo 注意: 训练过程中请勿关闭窗口，预计需要 1-3 小时 (取决于 GPU 性能)
echo.
python scripts/train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --data_dir ./processed_data --output_dir ./output/emotional_qwen
if %errorlevel% neq 0 (
    echo [错误] 训练失败
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 训练完成！
echo.
echo 使用以下命令进行测试:
echo   python scripts/inference.py --model_path ./output/emotional_qwen/final_model --mode interactive
echo.
echo 或运行批量测试:
echo   python scripts/inference.py --model_path ./output/emotional_qwen/final_model --mode batch
echo ============================================================
pause