from mmengine.analysis import get_model_complexity_info
from original_config.mmaction.models.backbones.timesformer import TimeSformer
x=(2,3,8,224,224)
model=TimeSformer(num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)).cuda()

analysis_results=get_model_complexity_info(model,x)
print("Model Flops:{}".format(analysis_results['flops_str']))
print("Model Parameters:{}".format(analysis_results['params_str']))
