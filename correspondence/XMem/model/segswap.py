import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from model import transformer


def get_model(model_path, eval_mode=False):
    ## set gpu
    device = torch.device("cuda")

    ## load Moco feature
    backbone = models.resnet50(pretrained=False)
    resnet_feature_layers = [
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
    ]
    resnet_module_list = [getattr(backbone, l) for l in resnet_feature_layers]
    last_layer_idx = resnet_feature_layers.index("layer3")
    backbone = torch.nn.Sequential(*resnet_module_list[: last_layer_idx + 1])

    ## load pre-trained weight
    resume_path_segswap = model_path  #'model/hard_mining_neg5.pth'
    pos_weight = 0.1
    feat_weight = 1
    dropout = 0.1
    activation = "relu"
    mode = "small"
    layer_type = ["I", "C", "I", "C", "I", "N"]
    drop_feat = 0.1
    feat_dim = 1024

    ## model
    netEncoder = transformer.TransEncoder(
        feat_dim,
        pos_weight=pos_weight,
        feat_weight=feat_weight,
        dropout=dropout,
        activation=activation,
        mode=mode,
        layer_type=layer_type,
        drop_feat=drop_feat,
    )

    netEncoder.cuda()

    if not eval_mode:
        print("Loading net weight from {}".format(resume_path_segswap))
        param = torch.load(resume_path_segswap)
        backbone.load_state_dict(param["backbone"])
        netEncoder.load_state_dict(param["encoder"], strict=False)
        # backbone.train()
        netEncoder.train()
    backbone.eval()
    # netEncoder.eval()
    netEncoder.cuda()

    for param in backbone.parameters():
        param.requires_grad = False
    for param in netEncoder.net.cls_branch.parameters():
        param.requires_grad = False

    return backbone, netEncoder


def get_tensors(I1np, I2np, M1np):
    img_size = 1120

    # masking Image1
    # M1np = (M1np > 0).astype(np.uint8)
    I1np = I1np  # * M1np[..., None]

    I1 = I1np  # Image.fromarray(I1np)
    I2 = I2np  # Image.fromarray(I2np)

    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    )
    tensor1 = I1  # resize(I1, img_size)
    tensor2 = I2  # resize(I2, img_size)
    tensor3 = M1np

    tensor1 = transformINet(tensor1).unsqueeze(0).cuda()
    tensor2 = transformINet(tensor2).unsqueeze(0).cuda()
    tensor3 = torch.from_numpy(tensor3).unsqueeze(0).type(torch.FloatTensor).cuda()

    return I1, I2, tensor1, tensor2, tensor3


def forward_pass(backbone, netEncoder, tensor1, tensor2, tensor3):
    # b, c = tensor4.size()[:2]
    # h, w = tensor4.size()[-2:]
    # h = int(h / 16)
    # w = int(w / 16)
    # img_resize = transforms.Compose([transforms.Resize(torch.Size([h, w]))])
    # tmp_tensor4 = img_resize(tensor4)
    # random_mask2 = torch.BoolTensor(torch.Size([b, c, h, w])).fill_(False)
    # random_mask1 = torch.BoolTensor(torch.Size([b, c, h, w])).fill_(False)

    # if torch.rand(1).item() > 0.5 and tmp_tensor4.sum() > 0:
    #     random_mask2 = tmp_tensor4 < 0.5

    # random_mask1 = random_mask1.cuda()
    # random_mask2 = random_mask2.cuda()

    with torch.no_grad():
        feat1 = backbone(tensor1)  ## feature
        feat1 = F.normalize(feat1, dim=1)  ## l2 normalization
        feat2 = backbone(tensor2)  ## features
        feat2 = F.normalize(feat2, dim=1)  ## l2 normalization
        # import pdb; pdb.set_trace()
        fmask = backbone(tensor3.repeat(1, 3, 1, 1))
        fmask = F.normalize(fmask, dim=1)

    # RX = torch.BoolTensor((1, 1, 30, 30)).fill_(False)
    # RY = torch.BoolTensor((1, 1, 30, 30)).fill_(False)
    out1, out2, featx, featy, out_cls = netEncoder(feat1, feat2, fmask)  ## predictions

    # visualize(tensor1, tensor2, tensor3, tensor4)

    return out1.narrow(1, 2, 1), out2.narrow(1, 2, 1), featx, featy, out_cls


def visualize(tensor1, tensor2, tensor3, tensor4):
    from PIL import Image

    b = tensor1.size()[0]

    for i in range(b):
        tensor1_i = tensor1[i].detach().cpu().numpy()
        tensor1_i = np.transpose(tensor1_i, (1, 2, 0)) * 255
        img_tensor1_i = Image.fromarray(tensor1_i.astype(np.uint8))

        tensor2_i = tensor2[i].detach().cpu().numpy()
        tensor2_i = np.transpose(tensor2_i, (1, 2, 0)) * 255
        img_tensor2_i = Image.fromarray(tensor2_i.astype(np.uint8))

        tensor3_i = tensor3[i].detach().cpu().numpy()
        tensor3_i = np.transpose(tensor3_i, (1, 2, 0)) * 255
        img_tensor3_i = Image.fromarray(
            tensor3_i.reshape(tensor3_i.shape[:2]).astype(np.uint8)
        )

        tensor4_i = tensor4[i].detach().cpu().numpy()
        tensor4_i = np.transpose(tensor4_i, (1, 2, 0)) * 255
        img_tensor4_i = Image.fromarray(
            tensor4_i.reshape(tensor4_i.shape[:2]).astype(np.uint8)
        )
