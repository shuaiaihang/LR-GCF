import torch
import torch.nn as nn
from codes.models.swin_decoder import SwinTransDecoder
from codes.models._base import BaseModel2D
from segmentation_models_pytorch.encoders import get_encoder


class SimpleRotationHead(nn.Module):
    """
    简单的旋转预测头：预测8×8分块的旋转角度
    使用一个全连接层直接连接特征到输出
    """
    def __init__(self, in_channels, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_rotations = 4  # 预测0°, 90°, 180°, 270°
        
        # 特征降维
        self.reduce = nn.Conv2d(in_channels, 128, kernel_size=1)
        
        # 全连接层 - 直接预测
        self.fc = nn.Linear(128 * patch_size * patch_size, patch_size * patch_size * self.num_rotations)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        
        # 降维
        x = self.reduce(x)  # [B, 128, 8, 8]
        # 展平
        x = x.view(B, -1)  # [B, 128*8*8]
        
        # 全连接预测
        logits = self.fc(x)  # [B, 8*8*4]
        
        # 重塑为 [B, 4, 7, 7]
        logits = logits.view(B, self.num_rotations, self.patch_size, self.patch_size)
        
        return logits

class UNetTF(BaseModel2D):

    def __init__(self,
                 encoder_name="resnet50",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 decoder_attention_type=None,
                 in_channels=3,
                 classes=2,
                 activation=None,
                 embed_dim=96,
                 norm_layer=nn.LayerNorm,
                 img_size=256,
                 patch_size=4,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=8,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 use_checkpoint=False,
                 ape=True,
                 cls=True,
                 contrast_embed=False,
                 contrast_embed_dim=256,
                 contrast_embed_index=-3,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 final_upsample="expand_first",
                 patches_resolution=[64, 64],
                 reduction_dim =128
                 ):
        super().__init__()


        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels
        

        
        encoder_out_channels = encoder_channels[-1]

        self.swin_decoder1 = SwinTransDecoder(classes, embed_dim, norm_layer, img_size, patch_size, depths, num_heads,
                                             window_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate, use_checkpoint,
                                             ape, mlp_ratio, drop_path_rate, final_upsample, patches_resolution,
                                             encoder_channels)

        self.swin_decoder2 = SwinTransDecoder(classes, embed_dim, norm_layer, img_size, patch_size, depths, num_heads,
                                             window_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate, use_checkpoint,
                                             ape, mlp_ratio, drop_path_rate, final_upsample, patches_resolution,
                                             encoder_channels)
        





        # 使用简单的旋转头
        if cls:
            self.rotation_head = SimpleRotationHead(
                in_channels=encoder_out_channels,
                patch_size=8  # 8×8分块
            )
        else:
            self.rotation_head = None

        
        
        
    def forward(self, x, device=None):
        features = self.encoder(x)                     # 编码器多层输出
        
        # 两个decoder同时输出
        seg_tf1, attn_list1, _ = self.swin_decoder1(features, device)  # 第一个decoder
        seg_tf2, attn_list2, _ = self.swin_decoder2(features, device)  # 第二个decoder

        feat = features[-1]
        
        # #添加旋转预测
        if self.rotation_head is not None:
            rotation_logits = self.rotation_head(feat)  # [B, 4, 7, 7]
        else:
            rotation_logits = None
        out = {
            'seg_tf1': seg_tf1,
            'seg_tf2': seg_tf2,
            'rotation_logits': rotation_logits,  # [B, 4, 8, 8]
            'attn_list1': attn_list1,
            'attn_list2': attn_list2,
            'features': features,  # 编码器各层特征
        }
        return out
    def inference_tf(self, x, device, **kwargs):
        features = self.encoder(x)
        
        # 获取两个decoder的输出
        seg_tf1, _, _ = self.swin_decoder1(features, device)
        seg_tf2, _, _ = self.swin_decoder2(features, device)
        
        # 对两个decoder的输出取平均（在softmax空间）
        seg_tf1_prob = torch.softmax(seg_tf1, dim=1)
        seg_tf2_prob = torch.softmax(seg_tf2, dim=1)
        avg_prob = (seg_tf1_prob + seg_tf2_prob) / 2.0
        
        # 取argmax得到最终预测
        preds = torch.argmax(avg_prob, dim=1, keepdim=True).to(torch.float)
        return preds
    
    def inference_tf_1(self, x, device, **kwargs):
        features = self.encoder(x)
        seg_tf, _, _ = self.swin_decoder1(features, device)
        preds = torch.argmax(seg_tf, dim=1, keepdim=True).to(torch.float)

        return preds

    def inference_tf_2(self, x, device, **kwargs):
        features = self.encoder(x)
        seg_tf, _, _ = self.swin_decoder2(features, device)
        preds = torch.argmax(seg_tf, dim=1, keepdim=True).to(torch.float)
        return preds
