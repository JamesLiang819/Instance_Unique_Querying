_base_ = ['./mask2former_r50_lsj_8x2_100e_coco_contrast.py']
pretrained = 'swin_base_patch4_window12_384_22k.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    type='Mask2Former_contrast',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(
        type='Mask2FormerHead_contrast', in_channels=[128, 256, 512, 1024]),
    init_cfg=None)

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     weight_decay=0.0001,
#     eps=1e-8,
#     betas=(0.9, 0.999),
#     paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     gamma=0.1,
#     by_epoch=False,
#     step=[655556, 710184],
#     warmup='linear',
#     warmup_by_epoch=False,
#     warmup_ratio=1.0,  # no warmup
#     warmup_iters=1000)
