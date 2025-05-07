import os
import pandas as pd

def load_dataset(data_root='data', train=True):
    """
    读取整个数据集，返回一个包含所有样本信息的列表。
    每个样本是一个字典，包含：
    - 'id': 样本ID（字符串）
    - 'captcha_path': 验证码图片路径
    - 'single_char_paths': 单字图片路径列表（长度为9）
    - 'label': 验证码中4个字符对应的单字索引（字符串）
    """
    # 根据 train 参数选择子目录和标签文件
    subset = 'train' if train else 'test'
    label_file = 'train_label.txt' if train else 'test_label.txt'

    dataset = []
    label_df = pd.read_csv(os.path.join(data_root, label_file), sep=' ', header=None, names=['id', 'label'])

    for _, row in label_df.iterrows():
        # 修改：正确解析 id 和 label
        sample_id, label = row['id'].split(',', 1)  # 逗号前为 id，逗号后为 label
        sample_dir = os.path.join(data_root, subset, sample_id)
        captcha_path = os.path.join(sample_dir, f'{sample_id}.png')
        single_char_paths = [os.path.join(sample_dir, f'{i}.png') for i in range(9)]

        dataset.append({
            'id': sample_id,
            'captcha_path': captcha_path,
            'single_char_paths': single_char_paths,
            'label': label
        })

    return dataset