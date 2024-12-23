import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image
import io
import numpy as np
# 指定要读取的原始 TensorBoard 日志目录
original_log_dir = '/home/qingyuhan/sota_code/mmaction2/vis_with_dropout_background'
new_log_dir = '/home/qingyuhan/sota_code/mmaction2/out_vis_with_dropout_background'

# 创建一个新的 SummaryWriter 用于保存过滤后的数据
writer = SummaryWriter(new_log_dir)

# 加载 TensorBoard 数据
ea = event_accumulator.EventAccumulator(original_log_dir)
ea.Reload()  # 加载事件数据

# 获取所有图片数据（你也可以根据需要获取标量、直方图等数据）
tags = ea.Tags()['images']

# 过滤出以 'Val_15' 开头的 tags
filtered_tags = [tag for tag in tags if tag.startswith('Val_18')]

# 遍历过滤后的 tags，提取并写入新的 TensorBoard 文件
for tag in filtered_tags:
    events = ea.Images(tag)  # 获取该 tag 下的所有图片事件
    for event in events:
        # 将 encoded_image_string 转换为 PIL 图像
        img = Image.open(io.BytesIO(event.encoded_image_string))

        # 将 PIL 图像转换为张量 (C, H, W)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0  # 转换为 0-1 之间的张量

        # 将图片数据写入到新的 TensorBoard 文件
        writer.add_image(tag, img_tensor, global_step=event.step)

# 关闭 writer，保存到新的日志文件
writer.close()
