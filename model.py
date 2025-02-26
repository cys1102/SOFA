import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
    
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, feat_pre, feat_feat):
   
        B, C, H, W = feat_pre.size()
        N = H * W
        pre_flat = feat_pre.view(B, C, N).permute(0, 2, 1)   
        feat_flat = feat_feat.view(B, C, N).permute(0, 2, 1) 

        queries = self.query_proj(pre_flat)
        keys = self.key_proj(feat_flat)
        values = self.value_proj(feat_flat)
        attn_output, _ = self.attn(query=queries, key=keys, value=values)
        fused_flat = self.out_proj(attn_output)  
        fused = fused_flat.permute(0, 2, 1).view(B, C, H, W)
        return fused


class SOFANet(nn.Module):
    def __init__(self, out_channels, num_views, freeze_encoder=False):
    
        super(SOFANet, self).__init__()
        self.num_views = num_views
        
        # Encoder for pre-ablation images (3 channels)
        self.pre_down1 = DoubleConv(3, 64)
        self.pre_pool1 = nn.MaxPool2d(2)
        self.pre_down2 = DoubleConv(64, 128)
        self.pre_pool2 = nn.MaxPool2d(2)
        self.pre_down3 = DoubleConv(128, 256)
        self.pre_pool3 = nn.MaxPool2d(2)
        self.pre_down4 = DoubleConv(256, 512)
        self.pre_pool4 = nn.MaxPool2d(2)
        
        # Encoder for ablation features (4 channels)
        self.feat_down1 = DoubleConv(4, 64)
        self.feat_pool1 = nn.MaxPool2d(2)
        self.feat_down2 = DoubleConv(64, 128)
        self.feat_pool2 = nn.MaxPool2d(2)
        self.feat_down3 = DoubleConv(128, 256)
        self.feat_pool3 = nn.MaxPool2d(2)
        self.feat_down4 = DoubleConv(256, 512)
        self.feat_pool4 = nn.MaxPool2d(2)
        
        self.pre_bottleneck = DoubleConv(512, 1024)
        self.feat_bottleneck = DoubleConv(512, 1024)
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(embed_dim=1024, num_heads=8)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(out_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        
        self.classifier_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.BatchNorm1d(1024),
                    nn.Linear(1024, 1),
                )
                for _ in range(num_views)
            ]
        )
        
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        encoder_layers = [
            self.pre_down1, self.pre_down2, self.pre_down3, self.pre_down4,
            self.feat_down1, self.feat_down2, self.feat_down3, self.feat_down4,
            self.pre_bottleneck, self.feat_bottleneck
        ]
        for layer in encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def encode(self, x):
        pre = x[:, :3, :, :]    
        feat = x[:, 3:, :, :]   
        
        d1_pre = self.pre_down1(pre)
        p1_pre = self.pre_pool1(d1_pre)
        d2_pre = self.pre_down2(p1_pre)
        p2_pre = self.pre_pool2(d2_pre)
        d3_pre = self.pre_down3(p2_pre)
        p3_pre = self.pre_pool3(d3_pre)
        d4_pre = self.pre_down4(p3_pre)
        p4_pre = self.pre_pool4(d4_pre)
        bn_pre = self.pre_bottleneck(p4_pre) 
        
        d1_feat = self.feat_down1(feat)
        p1_feat = self.feat_pool1(d1_feat)
        d2_feat = self.feat_down2(p1_feat)
        p2_feat = self.feat_pool2(d2_feat)
        d3_feat = self.feat_down3(p2_feat)
        p3_feat = self.feat_pool3(d3_feat)
        d4_feat = self.feat_down4(p3_feat)
        p4_feat = self.feat_pool4(d4_feat)
        bn_feat = self.feat_bottleneck(p4_feat)  

        fused_bn = self.fusion(bn_pre, bn_feat)  
        return fused_bn, (d1_pre, d2_pre, d3_pre, d4_pre) 

    def decode(self, bn, encoder_feats):
        d1, d2, d3, d4 = encoder_feats
        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)
        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        out = self.final_conv(u1)
        out = torch.sigmoid(out)
        mask_out = self.mask_head(out)
        mask_out = torch.sigmoid(mask_out)
        return out, mask_out

    def forward(self, x):
        if x.dim() == 4:
            fused_bn, encoder_feats = self.encode(x)
            sim_out, mask_out = self.decode(fused_bn, encoder_feats)
            cls_logit = self.classifier_heads[0](fused_bn)
            cls_logit = cls_logit.unsqueeze(1)  
            outcome_pred = cls_logit.mean(dim=1)  
            return sim_out, outcome_pred, mask_out
        else:
            B, N, _, H, W = x.shape
            sim_outputs = []
            cls_logits = []
            mask_outs = []
            for i in range(N):
                xi = x[:, i, :, :, :]  
                fused_bn, encoder_feats = self.encode(xi)
                sim_out, mask_out = self.decode(fused_bn, encoder_feats)
                sim_outputs.append(sim_out)
                mask_outs.append(mask_out)
                logit = self.classifier_heads[i](fused_bn)  
                cls_logits.append(logit)
            sim_outputs = torch.stack(sim_outputs, dim=1) 
            cls_logits = torch.stack(cls_logits, dim=1)     
            outcome_pred = cls_logits.mean(dim=1)            
            return sim_outputs, outcome_pred, mask_outs