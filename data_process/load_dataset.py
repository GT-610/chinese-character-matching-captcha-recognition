import os
import pandas as pd

def load_dataset(data_root='data', train=True):
    """
    读取整个数据集，返回一个包含所有样本信息的列表。
    每个样本是一个字典，包含：
    - 'id': 样本ID（字符串）
    - 'captcha_path': 验证码图片路径
    - 'label': 验证码中4个字符对应的真实字符序列（字符串）
    """
    # 根据 train 参数选择子目录和标签文件
    subset = 'train' if train else 'test'
    label_file = 'train_label.txt' if train else 'test_label.txt'

    dataset = []
    label_df = pd.read_csv(os.path.join(data_root, label_file), sep=' ', header=None, names=['id', 'label'])

    for _, row in label_df.iterrows():
        parts = row['id'].split(',', 1)
        if len(parts) != 2:
            continue  # 跳过格式错误的行
            
        sample_id, label = parts
        sample_dir = os.path.join(data_root, subset, sample_id)
        captcha_path = os.path.join(sample_dir)

        dataset.append({
            'id': sample_id,
            'captcha_path': captcha_path,
            'label': label
        })

    return dataset