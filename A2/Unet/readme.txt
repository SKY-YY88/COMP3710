1、keras_png_slices_data
    训练数据

2.1、model（全量测试数据）
    Unet_D_30_16 模型文件路径
    gen_images 测试集推理结果
    pred_threshold 测试集推理结果（读取gen_images结果，做阈值处理）
    label_threshold 测试集原始标签

2.2 model（简单测试数据dsc>0.9）
    Unet_D_30_16 模型文件路径
    gen_images 测试集推理结果
    pred_threshold 测试集推理结果（读取gen_images结果，做阈值处理）
    label_threshold 测试集原始标签

3、Data_Loader.py 模型数据读取文件，使用单通道输入

4、losses.py 损失函数

5、Metrics.py 评测函数

6、train_eval.py 训练代码，python train_eval.py运行

7、test.py 全量测试数据集，测试代码 python test.py运行
    Dice Score : 0.8828969541883205

7、test_one.py 简单样本测试数据集（dsc>0.9）,测试代码 python test.py运行
    Dice Score : 0.9283340652395511

8、environment.yaml conda环境配置 Python 3.9.17