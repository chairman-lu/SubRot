import argparse, os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd.graph import save_on_cpu
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from vlmrot import data_utils, model_utils, flatquant_utils, utils

            
def vis_values(values, save_path):
    values = values.detach().cpu().numpy()
    x = np.arange(len(values))
    plt.figure(figsize=(12, 4))
    plt.plot(x, values)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close

def select_sensitive_subspace(values, ratio=0.8):
    """
    values:  [d]，torch.linalg.eigh返回的升序特征值
    vectors: [d, d]，特征向量位于列方向
    """
    values = values.clamp_min(0)

    # 从大到小排序
    values_desc, _ = torch.sort(values, descending=True)
    # 累计特征值贡献率
    cumulative_ratio = torch.cumsum(values_desc, dim=0) / values_desc.sum()

    # 达到ratio所需的最小方向数
    rank = int(
        torch.searchsorted(
            cumulative_ratio,
            torch.tensor(ratio, device=values.device),
        ).item()
    ) + 1

    return rank

def fit_loss_subspaces(model, image_processor, fisher_samples):
    layers = model_utils.get_layers(model)
    fisher_sums = [None] * len(layers)
    token_count = [0] * len(layers)
    def make_backward_hook(layer_id):
        def hook(_module, _grad_input, grad_output):
            grad = grad_output[0][0].float()   
            assert grad.ndim == 2   # NC
            # print(layer_id, grad.shape)
            gram = grad.T @ grad
            fisher_sums[layer_id] = (gram if fisher_sums[layer_id] is None else fisher_sums[layer_id] + gram)
            token_count[layer_id] += grad.shape[0]
        return hook

    hooks = []
    hooks.extend(layer.register_full_backward_hook(make_backward_hook(index))
                 for index, layer in enumerate(layers))
    with torch.enable_grad():
        for sample in tqdm(fisher_samples):
            model.zero_grad()
            input = model_utils.build_loss_inputs(sample, image_processor, model)
            # with save_on_cpu(pin_memory=True):
            output = model(**input, use_cache=False)
            loss = output.loss
            loss.backward()
    
    for hook in hooks:
        hook.remove()
    fishers = []
    ranks = []
    for i in range(len(layers)):
        fisher = fisher_sums[i]
        count = token_count[i]
        fisher = fisher / count
        fisher = (fisher + fisher.T) * 0.5
        assert torch.isfinite(fisher).all()

        values, vectors = torch.linalg.eigh(fisher)
        # vis_values(values, f'./cache/subspace/layer{i}_values.png')
        rank = select_sensitive_subspace(values)
        width = fisher.shape[0]
        indices = torch.cat((
            torch.arange(width - 1, width - rank - 1, -1),
            torch.arange(15, -1, -1),
        )).to(values.device)    # 取高rank个和低rank个, [16high, 16low]
        selected_values = values[indices].clamp_min_(0).cpu()
        basis = vectors[:, indices].float().cpu()
        fishers.append({'U': basis, 'values': selected_values}) 
        ranks.append(rank)
    return fishers, ranks

def fit_signed_a(model, image_processor, fisher, ranks, fisher_samples):
    # U.shape: (C, r)
    layers = model_utils.get_layers(model)
    a_list = [[] for _ in range(len(layers))]
    def make_backward_hook(layer_id):
        def hook(_module, _grad_input, grad_output):
            grad = grad_output[0][0].float()   
            assert grad.ndim == 2   # NC
            # print(layer_id, grad.shape)
            U = fisher[layer_id]['U'][:, :ranks[layer_id]].to(grad.device)
            projected_grad = grad.to(U.dtype) @ U   # Nr
            a_sample = projected_grad.sum(dim=0)    # r
            a_list[layer_id].append(a_sample)
        return hook

    hooks = []
    hooks.extend(layer.register_full_backward_hook(make_backward_hook(index))
                 for index, layer in enumerate(layers))
    with torch.enable_grad():
        for sample in tqdm(fisher_samples):
            model.zero_grad()
            input = model_utils.build_loss_inputs(sample, image_processor, model)
            # with save_on_cpu(pin_memory=True):
            output = model(**input, use_cache=False)
            loss = output.loss
            loss.backward()
    for hook in hooks:
        hook.remove()

    res = []
    for i in range(len(layers)):
        a = a_list[i]
        a = torch.stack(a)      # Br
        neg_sum = (a < -1e-8).sum(dim=0)    # r
        pos_sum = (a > 1e-8).sum(dim=0)
        neg_ratio = neg_sum / a.shape[0]
        pos_ratio = pos_sum / a.shape[0]
        res_i = torch.zeros_like(neg_ratio)     # r
        res_i[neg_ratio > 0.9] = -1
        res_i[pos_ratio > 0.9] = 1
        res.append(res_i)
        print(f'layer {i}, rank {a.shape[-1]}, '
              f'neg num {(res_i == -1).sum()}, pos num {(res_i == 1).sum()}, zero num {(res_i == 0).sum()}')
    return res

def get_unstable_r(a_matrices):
    unstable_mask = []
    for i in range(len(a_matrices)):
        a = a_matrices[i]   # r
        mask = torch.zeros_like(a).to(a)
        mask[a == 0] = 1
        unstable_mask.append(mask)
    return unstable_mask

def pad_and_stack(x1, x2):
    if x1.ndim == 2:
        # x1.shape: (N1, C), x2.shape: (N2, C), pad and stack to (2, max(N1, N2), C)
        max_len = max(x1.shape[0], x2.shape[0])
        # (C左, C右, N左, N右)
        x1 = F.pad(x1, (0, 0, 0, max_len - x1.shape[0]))
        x2 = F.pad(x2, (0, 0, 0, max_len - x2.shape[0]))
        return torch.stack([x1, x2])
    elif x1.ndim == 3:
        # x1.shape: (B, N1, C), x2.shape: (N2, C), pad and stack to (B + 1, max(N1, N2), C)
        x2 = x2.unsqueeze(0)  # (1, N2, C)
        max_len = max(x1.shape[1], x2.shape[1])
        x1 = F.pad(x1, (0, 0, 0, max_len - x1.shape[1]))
        x2 = F.pad(x2, (0, 0, 0, max_len - x2.shape[1]))
        return torch.cat([x1, x2])

def conservative_channel_sign(x, length, threshold=0.8):
    """
    x.shape: (B, N, C), x in {-1, 0, 1}
    length.shape: (B)
    return shape: C
        set value = -1 only if most valid x is -1 else 1 
    """
    B, N, C = x.shape
    assert B == length.shape[0]
    assert torch.all((x == 0) | (x == 1) | (x == -1))

    mask = torch.arange(N).unsqueeze(0) < length.unsqueeze(-1)
    mask = mask.unsqueeze(-1).to(x.device)     # B, N, 1
    neg_count = ((x == -1) & (mask)).sum(dim=(0, 1))    # C
    total_count = mask.sum(dim=(0, 1))
    neg_ratio = neg_count / total_count     # C
    channel_sign = torch.where(neg_ratio >= threshold, 
                               -torch.ones_like(neg_ratio), torch.ones_like(neg_ratio))
    return channel_sign
         
def subspace_similarity(U1, U2):
    """
    U1, U2: [r, d]，每一行是一个子空间方向
    """
    # 保证行向量构成标准正交基
    Q1, _ = torch.linalg.qr(U1.T.float(), mode="reduced")  # [d, r]
    Q2, _ = torch.linalg.qr(U2.T.float(), mode="reduced")  # [d, r]

    # 奇异值就是各主角度的 cos 值
    cos_theta = torch.linalg.svdvals(Q1.T @ Q2)
    cos_theta = cos_theta.clamp(0, 1)

    return {
        "mean_cosine": cos_theta.mean(),
        "mean_squared_cosine": cos_theta.square().mean()
    }

def _mean_overlap(old_basis, new_basis):
    if old_basis.shape[-1] != new_basis.shape[-1]:
        new_rank = min(old_basis.shape[-1], new_basis.shape[-1])
        old_basis = old_basis[:, :new_rank]
        new_basis = new_basis[:, :new_rank]
        # print('clipped rank!')
    singular = torch.linalg.svdvals(old_basis.double().T @ new_basis.double())
    singular = singular.clamp(0.0, 1.0)
    return float(singular.square().mean())

def weighted_subspace_similarity(U1, eig1, U2, eig2, eps=1e-12):
    """
    U1, U2: [r, d]，要求每行正交归一
    eig1, eig2: [r]
    """
    U1 = U1.float()
    U2 = U2.float()
    eig1 = eig1.float().clamp_min(0)
    eig2 = eig2.float().clamp_min(0)

    # C[i,j] = <u1_i, u2_j>
    C = U1.T @ U2  # [r, r]

    # <F1,F2>_F = sum_ij λ1_i λ2_j <u1_i,u2_j>²
    numerator = (
        eig1[:, None] * eig2[None, :] * C.square()
    ).sum()

    # ||U^T diag(λ) U||_F = sqrt(sum λ_i²)
    denominator = (
        eig1.square().sum().sqrt()
        * eig2.square().sum().sqrt()
    ).clamp_min(eps)

    return numerator / denominator

def make_subspace_noise_hook(U, noise_ratio=0.1, seed=42):
    # U.shape: C,r
    def hook(module, inputs, outputs):
        nonlocal U
        assert isinstance(outputs, tuple)
        hidden = outputs[0]
        # 做一个shape为BNr的噪声，将其加到子空间Cr上，再映射回BNC
        U = U.to(hidden)
        B, N, C = hidden.shape
        r = U.shape[1]
        assert U.shape[0] == C

        generator = torch.Generator(device=hidden.device)
        generator.manual_seed(seed)
        noise = torch.randn(B, N, r, device=hidden.device, dtype=hidden.dtype, generator=generator)
        noise = noise @ U.T

        hidden_rms = hidden.float().square().mean().sqrt()
        noise_rms = noise.square().mean().sqrt().clamp_min(1e-12)
        noise = noise * (noise_ratio * hidden_rms / noise_rms)

        noisy_hidden = hidden + noise
        return (noisy_hidden,) + outputs[1:]
    return hook

def make_signed_subspace_noise_hook(U, a, noise_ratio=0.1, seed=42):
    # U.shape: C, r, a.shape: (r)
    def hook(module, inputs, outputs):
        nonlocal U
        assert isinstance(outputs, tuple)
        hidden = outputs[0]
        # 做一个shape为BNr的噪声，将其加到子空间Cr上，再映射回BNC
        U = U.to(hidden)
        B, N, C = hidden.shape
        r = U.shape[1]
        assert U.shape[0] == C

        generator = torch.Generator(device=hidden.device)
        generator.manual_seed(seed)
        noise = -abs(torch.randn(1, 1, r, device=hidden.device, dtype=hidden.dtype, generator=generator))
        noise = noise.expand(B, N, r)
        mask = a.reshape(1, 1, -1).expand_as(noise).to(hidden)

        hidden_high = hidden @ U
        hidden_rms = hidden_high.square().mean().sqrt()
        noise_rms = noise.square().mean().sqrt()
        noise = noise * (noise_ratio * hidden_rms / noise_rms)

        noise = noise * mask
        noise = noise @ U.T
        noisy_hidden = hidden + noise
        return (noisy_hidden,) + outputs[1:]
    return hook


def get_effect_score(loss_ori: list, loss: list):
    assert len(loss_ori) == len(loss)
    effect = [abs(loss_ori[i] - loss[i]) for i in range(len(loss))]
    effect = sum(effect) / len(effect)
    return effect

def get_signed_effect_score(loss_ori: list, loss: list):
    assert len(loss_ori) == len(loss)
    effect = [loss[i] - loss_ori[i] for i in range(len(loss))]
    effect = sum(effect) / len(effect)
    return effect

def sample_forward(model, image_processor, samples):
    losses = []
    with torch.no_grad():
        for sample in samples:
            model.zero_grad()
            input = model_utils.build_loss_inputs(sample, image_processor, model)
            output = model(**input, use_cache=False)
            loss = output.loss
            losses.append(loss)
    return losses

def fit_margin(model, image_processor, fishers, ranks, a_matrices, fisher_samples):
    loss_ori = sample_forward(model, image_processor, fisher_samples)
    # 逐层扰动，观察最终 loss 变化
    layers = model_utils.get_layers(model)
    ratios = []
    for i in range(len(layers)):
        fisher = fishers[i]
        rank = ranks[i]
        a = a_matrices[i]

        noise_ratio = 0.001
        step1 = 0.05
        step2 = 0.005
        def search(init_ratio, step):
            noise_ratio = init_ratio
            signed_effect = 0
            loss_change = -1e5
            while True:
                hook = layers[i].register_forward_hook(make_signed_subspace_noise_hook(
                                fisher['U'][:, :rank], a, noise_ratio=noise_ratio))
                loss_new = sample_forward(model, image_processor, fisher_samples)
                new_signed_effect = get_signed_effect_score(loss_ori, loss_new)
                loss_change = new_signed_effect - signed_effect
                signed_effect = new_signed_effect
                hook.remove()
                if loss_change < 0:
                    noise_ratio += step
                else:
                    noise_ratio -= step
                    break
            return noise_ratio
        noise_ratio = search(noise_ratio, step1)
        noise_ratio = search(noise_ratio, step2)
        if noise_ratio < 0:
            print(f'current ratio {noise_ratio}, resetting to {ratios[-1]}')
            noise_ratio = ratios[-1]
        ratios.append(noise_ratio)
        print(f'layer {i}, ratio {noise_ratio}')
    return ratios

def main(args):
    # 1.保存8个样本用于计算子空间
    model, image_processor = model_utils.get_model(args.model, args.model_path)
    model.eval()
    dataloader = data_utils.get_loaders(args.cal_dataset, 512, args.seed, question_file=args.question_file, image_base=args.image_base)
    fisher_samples1 = dataloader[:128]
    fisher_samples2 = dataloader[256:256 + 16]

    # 2.计算子空间并保存映射矩阵U
    # fisher1, ranks1 = fit_loss_subspaces(model, image_processor, fisher_samples1)
    # subspaces = {'fisher': fisher1, 'ranks': ranks1}
    # save_dir = Path(args.subspace_save_dir)
    # save_dir.mkdir(parents=True, exist_ok=True)
    # save_path = os.path.join(save_dir, 'subspace_on_ori.pt')
    # torch.save(subspaces, save_path)
    # print(f'Subpace on ori fp16 model saved to {save_path}')
    # fisher2, ranks2 = fit_loss_subspaces(model, image_processor, fisher_samples2)

    # # 3.验证输入样本对子空间的影响
    # for i, (f1, f2) in enumerate(zip(fisher1, fisher2)):
    #     U1, V1 = f1['U'][:, :ranks1[i]], f1['values'][:ranks1[i]]
    #     U2, V2 = f2['U'][:, :ranks2[i]], f2['values'][:ranks2[i]]
    #     mean_squared_cosine = _mean_overlap(U1, U2)
    #     weighted_score = weighted_subspace_similarity(U1, V1, U2, V2)
    #     print(f'layer: {i}, rank1: {ranks1[i]}, rank2: {ranks2[i]}, mean_squared_cosine: {mean_squared_cosine:.3f}, weight_score: {weighted_score:.3f}')



    # # 5.在一个训练好的Flatquant模型上取得敏感子空间
    # model = utils.move_model_to_cpu(model)
    # print(f'Using FlatQuant matrices from {args.flatquant_martix_path}')
    # flatquant_utils.apply_flatquant_to_model(args, model)
    # flatquant_utils.load_flat_matrices(args, model, args.flatquant_matrix_path)
    # flatquant_utils.reparameterize_model(model)
    # model = utils.dispatch_model_to_cuda(model)
    # fisher, ranks = fit_loss_subspaces(model, image_processor, fisher_samples1)
    # subspaces = {'fisher': fisher, 'ranks': ranks}
    # save_path = os.path.join(save_dir, 'subspace_on_flatquant.pt')
    # torch.save(subspaces, save_path)
    # print(f'Subpace on flatquant model saved to {save_path}')

    # fisher2 = fisher
    # ranks2 = ranks
    # # 3.验证输入样本对子空间的影响
    # for i, (f1, f2) in enumerate(zip(fisher1, fisher2)):
    #     U1, V1 = f1['U'][:, :ranks1[i]], f1['values'][:ranks1[i]]
    #     U2, V2 = f2['U'][:, :ranks2[i]], f2['values'][:ranks2[i]]
    #     mean_squared_cosine = _mean_overlap(U1, U2)
    #     weighted_score = weighted_subspace_similarity(U1, V1, U2, V2)
    #     print(f'layer: {i}, rank1: {ranks1[i]}, rank2: {ranks2[i]}, mean_squared_cosine: {mean_squared_cosine:.3f}, weight_score: {weighted_score:.3f}')
    
    # 6.考虑梯度正负号对最终 loss 的影响
    fishers, ranks = fit_loss_subspaces(model, image_processor, fisher_samples1)
    a = fit_signed_a(model, image_processor, fishers, ranks, fisher_samples1)
    # ratios = fit_margin(model, image_processor, fishers, ranks, a, fisher_samples2)
    # subspaces = {'fisher': fishers, 'ranks': ranks, 'a': a, 'ratios': ratios}
    subspaces = {'fisher': fishers, 'ranks': ranks, 'a': a}
    save_dir = Path(args.subspace_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model.split('/')[-1]
    save_path = os.path.join(save_dir, f'{model_name}_subspace_on_flatquant_with_a_128.pt')
    torch.save(subspaces, save_path)
    print(f'Subpace on flatquant model with "a" matrix saved to {save_path}')

    # 4.验证子空间对最终loss的影响
    utils.cleanup_memory()
    valid_samples = dataloader[-16:]
    loss_ori = sample_forward(model, image_processor, valid_samples)
    for i in range(len(fishers)):
        layer = model_utils.get_layers(model)[i]
        fisher = fishers[i]
        hook = layer.register_forward_hook(make_signed_subspace_noise_hook(fisher['U'][:, :ranks[i]], a[i], noise_ratio=0.1))
        loss_high = sample_forward(model, image_processor, valid_samples)
        effect_high = get_effect_score(loss_ori, loss_high)
        signed_high_effect = get_signed_effect_score(loss_ori, loss_high)
        hook.remove()

        print(f'layer {i}, high rank effect: {effect_high:.3f}, '
        f'signed high rank effect: {signed_high_effect:.3f}')

        # hook = layer.register_forward_hook(make_signed_subspace_noise_hook(fisher['U'][:, -16:], a[i]))
        # loss_low = sample_forward(model, image_processor, valid_samples)
        # effect_low = get_effect_score(loss_ori, loss_low)
        # signed_low_effect = get_signed_effect_score(loss_ori, loss_low)
        # hook.remove()
        # print(f'layer {i}, high rank effect: {effect_high:.3f}, low rank effect: {effect_low:.3f}, '
        #       f'signed high rank effect: {signed_high_effect:.3f}, signed low rank effect: {signed_low_effect:.3f}')

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_rank", type=int, default=16)
    parser.add_argument("--num_fisher_samples", type=int, default=8)
    parser.add_argument("--eval_layer", type=int, default=0)
    parser.add_argument("--flatquant_martix_path", type=str, default='cache/flatquant/llava-v1.6-vicuna-7b-hf/w4a4/passact2-15epoch/flat_matrices.pth')
    parser.add_argument("--subspace_save_dir", type=str, default='cache/subspaceloss')
    parser.add_argument("--cal_dataset", type=str, default='ScienceQA_Train')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question_file", type=str, default='/home/disk/workspace/QSVD/myllava/playground/data/eval/scienceqa/llava_train_QCM-LEA.json')
    parser.add_argument("--image_base", type=str, default='/home/disk/workspace/QSVD/myllava/playground/data/eval/scienceqa/images/train')
    parser.add_argument("--model", type=str, default='llava-hf/llava-v1.6-vicuna-7b-hf')
    parser.add_argument("--model_path", type=str, default='/home/disk/workspace/models/LLaVA-Next/llava-v1.6-vicuna-7b-hf')
    args = parser.parse_args()

    # Match the FlatQuant defaults from utils.parser_gen(). Config values below
    # override these defaults, including keys not declared by this local parser.
    flatquant_defaults = {
        "w_bits": 16,
        "a_bits": 16,
        "q_bits": 16,
        "k_bits": 16,
        "v_bits": 16,
        "w_asym": False,
        "a_asym": False,
        "q_asym": False,
        "k_asym": False,
        "v_asym": False,
        "a_groupsize": -1,
        "lwc": False,
        "lac": False,
        "add_diag": False,
        "diag_init": "sq_style",
        "diag_alpha": 0.3,
        "direct_inv": False,
        "separate_vtrans": False,
    }
    for key, value in flatquant_defaults.items():
        setattr(args, key, value)

    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        for key, value in config.items():
            setattr(args, key, value)
    utils.set_seed(args.seed)
    main(args)
