import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear

def mySigmoid(x,lamb):
    return (x/lamb).sigmoid()

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').view(-1)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances,contrast_mask_feats,contrast_instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        contrast_locations = compute_locations(
            contrast_mask_feats.size(2), contrast_mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)
        # n_inst = 500

        im_inds = instances.im_inds
        contrast_im_inds = contrast_instances.im_inds# contrast

        mask_head_params = instances.mask_head_params
        contrast_mask_head_params = contrast_instances.mask_head_params # contrast

        # print(mask_head_params.shape,contrast_mask_head_params.shape)
        N, _, H, W = mask_feats.size()
        N2, _, H2, W2 = contrast_mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            # contrast
        #     contrast_locations = compute_locations(
        #     contrast_mask_feats.size(2), contrast_mask_feats.size(3),
        #     stride=mask_feat_stride, device=contrast_mask_feats.device
        # )
            contrast_instance_locations = contrast_instances.locations
            contrast_relative_coords = contrast_instance_locations.reshape(-1, 1, 2) - contrast_locations.reshape(1, -1, 2)
            contrast_relative_coords = contrast_relative_coords.permute(0, 2, 1).float()
            contrast_soi = self.sizes_of_interest.float()[contrast_instances.fpn_levels]
            contrast_relative_coords = contrast_relative_coords / contrast_soi.reshape(-1, 1, 1)
            contrast_relative_coords = contrast_relative_coords.to(dtype=contrast_mask_feats.dtype)
            contrast_mask_head_inputs = torch.cat([
                contrast_relative_coords, contrast_mask_feats[contrast_im_inds].reshape(n_inst, self.in_channels, H2 * W2)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            # contrast_mask_head_inputs = torch.cat([
                # relative_coords, contrast_mask_feats[contrast_im_inds].reshape(n_inst, self.in_channels, H2 * W2)
            # ], dim=1)  
        # print(mask_head_inputs.shape,im_inds.shape,mask_feats.shape)
        # print('mask_logits',mask_head_inputs.shape)
        # print('contrast_mask_logits',contrast_mask_head_inputs.shape)
        # print(mask_head_params.shape,mask_head_inputs.shape,mask_feats.shape,self.channels)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        contrast_mask_head_inputs = contrast_mask_head_inputs.reshape(1, -1, H2, W2) # contrast

        # print(mask_head_params.shape,contrast_mask_head_params.shape)
        # print(mask_feats.shape,contrast_mask_feats.shape)
        # print(mask_head_inputs.shape,contrast_mask_head_inputs.shape)


        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )
        contrast_weights, contrast_biases = parse_dynamic_params(
            contrast_mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )
        # if base_weights!=None:
        #     print(self.channels)
        # contrast_weights, contrast_biases = parse_dynamic_params(
        #     contrast_mask_head_params, self.channels,
        #     self.weight_nums, self.bias_nums
        # )
        # print(mask_head_inputs.shape,mask_feats.shape)
        # raise NameError("STOP!")
        # mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        # aug_mask_logits = self.mask_heads_forward(contrast_mask_head_inputs, contrast_weights, contrast_biases, n_inst)

        contrast_mask_logits_1 = self.mask_heads_forward(contrast_mask_head_inputs, contrast_weights, contrast_biases, n_inst) # contrast
        contrast_mask_logits_2 = self.mask_heads_forward(contrast_mask_head_inputs.detach(), weights, biases, n_inst) # contrast
        # contrast_aug_mask_logits = self.mask_heads_forward(mask_head_inputs, contrast_weights, contrast_biases, n_inst)


        mask_logits = mask_logits.reshape(-1, 1, H, W)
        contrast_mask_logits_1 = contrast_mask_logits_1.reshape(-1, 1, H2, W2) # contrast
        contrast_mask_logits_2 = contrast_mask_logits_2.reshape(-1, 1, H2, W2) # contrast


        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        contrast_mask_logits_1 = aligned_bilinear(contrast_mask_logits_1, int(mask_feat_stride / self.mask_out_stride))
        contrast_mask_logits_2 = aligned_bilinear(contrast_mask_logits_2, int(mask_feat_stride / self.mask_out_stride))

        # mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride))
        # contrast_mask_logits = aligned_bilinear(contrast_mask_logits, int(mask_feat_stride))

        # aug_mask_logits = aligned_bilinear(aug_mask_logits, int(mask_feat_stride / self.mask_out_stride))
        # contrast_aug_mask_logits = aligned_bilinear(contrast_aug_mask_logits, int(mask_feat_stride / self.mask_out_stride))
        # print('mask_logits',mask_logits.shape)
        # print('contrast_mask_logits',contrast_mask_logits.shape)
        # # print("!!!!!")
        # raise NameError("STOP!")
        return mask_logits,contrast_mask_logits_1,contrast_mask_logits_2

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None,contrast_mask_feats=None,contrast_pred_instances=None,contrast_gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            contrast_gt_inds = contrast_pred_instances.gt_inds
            # contrast_gt_bitmasks = torch.cat([torch.flip(per_im.gt_bitmasks,dims=[2]) for per_im in gt_instances])
            contrast_gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in contrast_gt_instances])
            contrast_gt_bitmasks = contrast_gt_bitmasks[contrast_gt_inds].unsqueeze(dim=1).to(dtype=contrast_mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                
                mask_logits,contrast_mask_logits_1,contrast_mask_logits_2 = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances,contrast_mask_feats,contrast_pred_instances
                )
                # mask_scores = mask_logits.sigmoid()
                # contrast_mask_scores_1 = contrast_mask_logits_1.sigmoid()
                # contrast_mask_scores_2 = contrast_mask_logits_2.sigmoid()
                mask_scores = mySigmoid(mask_logits,3)
                contrast_mask_scores_1 = mySigmoid(contrast_mask_logits_1,3)
                contrast_mask_scores_2 = mySigmoid(contrast_mask_logits_2,3)

                if self.boxinst_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

                    # contrast_mask_losses = dice_coefficient(contrast_mask_scores, 0*gt_bitmasks)
                    # cri=WeightedFocalLoss()
                    # contrast_mask_losses=torch.abs(contrast_mask_scores-0*gt_bitmasks)
                    contrast_mask_losses_1=dice_coefficient(contrast_mask_scores_1, contrast_gt_bitmasks)
                    # contrast_mask_losses = cri(contrast_mask_scores,0*gt_bitmasks)
                    contrast_loss_mask_1 = contrast_mask_losses_1.mean()
                    losses["contrast_loss_mask"] = contrast_loss_mask_1

                    # contrast_mask_scores_2[contrast_mask_scores_2<0.99]=0.01
                    # losses["contrast_loss_mask_max"]=(torch.max(contrast_mask_scores_2,1).values/10).mean()
                    # losses["contrast_loss_mask_max"]=torch.max(contrast_mask_scores_2)/9
                    
                    zero= torch.zeros_like(contrast_mask_scores_2)
                    contrast_mask_scores_2=torch.where(contrast_mask_scores_2<0.80,zero,contrast_mask_scores_2)
                    losses["contrast_loss_mask_max"]=contrast_mask_scores_2.mean()*10

            return losses
        else:
            if len(pred_instances) > 0:
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                mask_logits,_,_ = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances,mask_feats,pred_instances
                )
                # pred_instances.pred_global_masks = mask_logits.sigmoid()
                pred_instances.pred_global_masks = mySigmoid(mask_logits,3)

            return pred_instances
