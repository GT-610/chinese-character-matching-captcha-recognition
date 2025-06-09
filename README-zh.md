[English](README.md)

# 中文验证码识别

这个项目是我《机器学习课程设计》的作业，作业选题是中文验证码图像识别。目的是开发一个能够准确识别和转录验证码中中文字符的机器学习模型。

现在课程已结束，我决定开源该项目，允许其他人学习并可能为项目做出贡献。

## 简介

该项目旨在解决中文验证码图像识别的问题，通过使用机器学习技术，我们旨在开发一个能够准确转录验证码中中文字符的健壮系统。

数据集描述在数据集压缩文件中的 `README.md` 中。

## 结构

- `data_process/`: 包含数据预处理脚本。
- `feature_extract`: 包含特征提取脚本。
- `models/`: 存储训练好的模型和模型架构文件。
- `experiments/`: 包含解决问题的方法。
- `plot_figures/`: 包含绘图脚本。
- `main.py`: 用于运行项目的主入口。

## 要求

- Python 3.10+（我使用的是 Python 3.12）
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Seaborn

## 安装

1. 克隆此仓库。
2. 建议先创建一个虚拟环境。
3. 运行 `pip install -r requirements.txt` 以安装所需的包。
4. 根据需要自行用注释方式调整 `main.py` 中的代码。
5. 运行 `python main.py`。

## 许可证
本项目采用 [MIT 许可证](LICENSE)。