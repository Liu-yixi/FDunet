import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from model_archs.models_v8 import GAP_net, AttentionPooling
import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/qqshen/lowrankProject')
from VMamba.classification.models.vmamba import VSSM
from MambaIR.analysis.model_zoo.mambaIR import SS2D
if __name__ == "__main__":
    from utils import gen_meas_torch
    # net = GAP_net().cuda()
    # net = AttentionPooling(28, 28, 2).cuda()
    tens = torch.randn(5, 28, 256, 256).cuda()
    # model = VSSM(
    #     depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2, 
    #     patch_size=4, in_chans=28, num_classes=1000, 
    #     ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
    #     ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
    #     ssm_init="v2", forward_type="m0_noz", 
    #     mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
    #     patch_norm=True, norm_layer="ln",
    #     downsample_version="v3", patchembed_version="v2", 
    #     use_checkpoint=False, posembed=False, imgsize=256, 
    # )
    model = SS2D(d_model=28)
    model.cuda().train()
    # Phi_batch = torch.randn(5, 28, 256, 310).cuda()
    # Phi_s_batch = torch.sum(Phi_batch,1)
    # Phi_s_batch[Phi_s_batch==0] = 1
    # # y, PHi, Phi_s
    # y = gen_meas_torch(tens, Phi_batch, is_training=True)
    tens = tens.permute(0, 2, 3, 1)
    out = model(tens)
    print(out.shape)
    