import torch
import torchvision as tv
from torch import nn
from models.model.deformable_transformer import *
import numpy as np
from models.model.position_encoding import *
import cv2
import os


def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod

class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=1, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x):
        c1_ = (self.conv1(x)) # 200 48
        c1 = self.conv2(c1_) # 99 23 \ 99 46
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) # 31 6 \ 31 14
        c3 = self.conv3(v_max) # 31 6
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False) # 200 48
        cf = self.conv_f(c1_) # residual
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)

# Optional Loss
class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])
        bev_view = self.fc_transform(x)  # 拿出一个视角
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class CaliFree3DLane(nn.Module):
    def __init__(self, bev_shape, output_2d_shape, train=True):
        super(BEV_LaneDet, self).__init__()
        self.bb = nn.Sequential(*list(tv.models.resnet34(pretrained=True).children())[:-2])

        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)

                ),
                downsample=nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            )
        )
        self.down1 = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512)

                ),
                downsample=nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            )
        )

        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512)

        self.position_embedding = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
        self.query_embed = nn.Embedding(125, 1024) # 25*5*2
        self.transformer = DeformableTransformer(d_model=512, nhead=8,
                 num_encoder_layers=1, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_frames=2, num_feature_levels=2,
                 dec_n_points=4,  enc_n_points=4,)

        self.num_frames = 2 # temporal length

        # self.bev_compressor = nn.Sequential(
        #     nn.Conv2d(uv_feat_c * 3, uv_feat_c, kernel_size=3, padding=1, stride=1, bias=False),
        #     nn.InstanceNorm2d(uv_feat_c),
        #     nn.GELU(),
        # )

    def forward(self, img):
        # for i in range(img.shape[0]):
        #     img1 = img[i, ].permute(1, 2, 0).detach().cpu().numpy()
        #     if os.path.exists(f'bb_input_{i}.jpg'):
        #         cv2.imwrite(f'next_bb_input_{i}.jpg', img1)
        #     else:
        #         cv2.imwrite(f'bb_input_{i}.jpg', img1)
        img_s32 = self.bb(img) # bs*2(time) 3 576 1024 -> 6 3 576 1024
        n, c, h, w = img_s32.shape
        img_s32_input = img_s32.reshape(n // self.num_frames, self.num_frames, c, h, w)
        img_s32_input = img_s32_input[:, -1]
        img_s64 = self.down1(img_s32) # 8 1024 9 16 (8 512 9 16)
        srcs = [img_s32, img_s64]
        input_srcs = []
        masks = []
        poses = []
        for i, src in enumerate(srcs):
            mask = self.mask_generate(src)
            pose = self.position_embedding(src, mask)

            n, c, h, w = src.shape
            src = src.reshape(n // self.num_frames, self.num_frames, c, h, w)
            # print(src[2, ].eq(src1[1, 0, ])) # True
            mask = mask.reshape(n // self.num_frames, self.num_frames, h, w)
            pos = pose.reshape(n // self.num_frames, self.num_frames, c, h, w)
            input_srcs.append(src)
            masks.append(mask)
            poses.append(pos)

        query_embeds = self.query_embed.weight
        hs, hs_box, memory, init_reference, inter_references, inter_samples, enc_outputs_class, valid_ratios = self.transformer(
            input_srcs, masks, poses, query_embeds)
        bev = hs.permute(0, 2, 1).view(n // self.num_frames, c, 25, 5)

        if self.is_train:
            return self.lane_head(bev), self.lane_head_2d(img_s32_input)
        else:
            return self.lane_head(bev)

    def mask_generate(self, src):

        mask = np.ones((src.shape[0], src.shape[-2], src.shape[-1]), dtype=np.bool)
        mask = np.logical_not(mask)
        mask = torch.from_numpy(mask).cuda()
        return mask
