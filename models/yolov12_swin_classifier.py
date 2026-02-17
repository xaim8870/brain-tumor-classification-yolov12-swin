#models/yolov12_swin_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Swin Transformer backbone (hierarchical features)
class SwinBackbone(nn.Module):
    def __init__(self, variant='swin_tiny_patch4_window7_224', in_chans=3, pretrained=True):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=pretrained, features_only=True,
                                       in_chans=in_chans, out_indices=(1,2,3), img_size=512, window_size=8)
        self.channels = self.model.feature_info.channels()  # [C3,C4,C5] channels

    def forward(self, x):
        # returns list of feature maps [P3, P4, P5] in (B,C,H,W)
        feats = self.model(x)
        return [f.permute(0,3,1,2).contiguous() for f in feats]

# YOLO-inspired neck (region/self-attention blocks could be added here if needed)
class FPNNeck(nn.Module):
    def __init__(self, in_channels, out_c=256):
        super().__init__()
        c3, c4, c5 = in_channels
        # Reduce all to common channel size
        self.reduce3 = nn.Conv2d(c3, out_c, 1)
        self.reduce4 = nn.Conv2d(c4, out_c, 1)
        self.reduce5 = nn.Conv2d(c5, out_c, 1)
        # Optional: further conv layers for fusion (e.g., RELAN or attention blocks)

    def forward(self, c3, c4, c5):
        p3 = self.reduce3(c3)
        p4 = self.reduce4(c4)
        p5 = self.reduce5(c5)
        # Upsample lower levels to match P5 spatial dims
        p3_up = F.interpolate(p3, size=p5.shape[-2:], mode='nearest')
        p4_up = F.interpolate(p4, size=p5.shape[-2:], mode='nearest')
        # Concatenate and fuse
        fused = torch.cat([p3_up, p4_up, p5], dim=1)  # (B,3*out_c,H5,W5)
        return fused

# Full model with classification head
class YOLOv12SwinClassifier(nn.Module):
    def __init__(self, num_classes=3, backbone_variant='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.backbone = SwinBackbone(
            variant=backbone_variant,
            in_chans=3,
            pretrained=True
        )

        self.neck = FPNNeck(self.backbone.channels, out_c=128)

        self.merge_conv = nn.Conv2d(128 * 3, 256, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ✅ ADD DROPOUT (critical)
        self.dropout = nn.Dropout(p=0.4)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        fused = self.neck(c3, c4, c5)
        merged = F.relu(self.merge_conv(fused))
        pooled = self.avgpool(merged).view(x.size(0), -1)
        pooled = self.dropout(pooled)  # ✅
        logits = self.fc(pooled)
        return logits
