from vlmrot import model_utils, utils, hadamard_utils, quant_utils
from vlmrot.model_utils import LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL
import torch, transformers
import inspect, logging, typing
import re, math
from tqdm import tqdm
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

logging.basicConfig(level=logging.INFO, format='%(message)s')

def untie_embeddings_and_lm_head(model):
    """Separate shared weights before norm fusion and independent rotations."""
    head = model_utils.get_lm_head(model)
    if not any(
        embedding.weight is head.weight
        or embedding.weight.data_ptr() == head.weight.data_ptr()
        for embedding in model_utils.get_embeddings(model)
    ):
        return
    head.weight = nn.Parameter(
        head.weight.detach().clone(), requires_grad=head.weight.requires_grad
    )
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config.tie_word_embeddings = False


def fuse_layernorm(model, fuse_module):
    if fuse_module == 'vit':
        layers = model_utils.get_vit_layers(model)
        if type(model) == LLAVA_HF or type(model) == LLAVA_NEXT_HF or type(model) == LLAVA_OV_HF or type(model) == QWEN2_VL:
            vit_model = model.vision_tower.vision_model
            vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
            for idx, layer in enumerate(layers):
                # print(inspect.getsource(layer.forward))
                fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj,
                                                   layer.self_attn.k_proj, layer.self_attn.v_proj])
                fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
                w, b = fold_mean_into_linear(layer.self_attn.out_proj)
                if idx < len(layers) - 1:
                    w, b = fold_mean_into_linear(layer.mlp.fc2)
                else:
                    logging.info(f"skip mlp mean in layer{idx}")

            model_utils.replace_modules(
                vit_model.encoder,
                torch.nn.LayerNorm,
                lambda _: model_utils.RMSN(model_utils.get_vit_config(model).hidden_size, eps=1e-5),
                replace_layers=False,
            )

    elif fuse_module == 'llm':
        untie_embeddings_and_lm_head(model)
        layers = model_utils.get_transformer_layers(model)
        pre_head_norm = model_utils.get_pre_head_layernorm(model)
        lm_head = model_utils.get_lm_head(model)
        fuse_ln_linear(pre_head_norm, [lm_head])
        for layer in layers:
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj,
                                                    layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])

        norm_type = model_utils.get_lm_norm_type(model)
        model_utils.replace_modules(
            model,
            norm_type,
            lambda _: model_utils.RMSN(model.config.text_config.hidden_size, eps=1e-5),
            replace_layers=False,
        )
    else:
        raise ValueError

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()

        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)

def fold_mean_into_linear(linear: torch.nn.Linear):
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    W_mean = W_.mean(dim=-2, keepdim=True)
    linear.weight.data = W_ - W_mean
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        b_mean = b_.mean()
        linear.bias.data = b_ - b_mean
        linear.bias.data = linear.bias.data.to(linear_dtype)
    return (W_mean, b_mean)

_SPINQUANT_R2_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.R2(?:\.weight)?$")


def load_optimized_R1_R2(path, device=None):
    checkpoint = torch.load(path, map_location="cpu")
    if "R1" not in checkpoint:
        raise KeyError(f"Rotation checkpoint {path} does not contain R1")
    device = device or utils.get_dev()
    R1 = checkpoint["R1"].to(dtype=torch.float64, device=device)
    r2_by_layer = {}
    for key, value in checkpoint.items():
        match = _SPINQUANT_R2_RE.search(key)
        if match is None:
            continue
        r2_by_layer[int(match.group(1))] = value.to(
            dtype=torch.float64,
            device=device,
        )
    return R1, r2_by_layer


def _linear_from_module(module):
    if isinstance(module, nn.Linear):
        return module
    wrapped = getattr(module, "module", None)
    if isinstance(wrapped, nn.Linear):
        return wrapped
    raise TypeError(f"Expected nn.Linear or wrapper with .module, got {type(module)}")


def _apply_head_rotation_to_linear(module, head_dim, rotation, output=False):
    linear = _linear_from_module(module)
    weight = linear.weight.data
    dtype = weight.dtype
    device = weight.device
    work = weight.to(dtype=torch.float64)
    rot = rotation.to(device=device, dtype=work.dtype)

    if output:
        work = work.t()
        shape = work.shape
        work = torch.matmul(
            work.reshape(-1, shape[-1] // head_dim, head_dim),
            rot,
        ).reshape(shape).t()
        if linear.bias is not None:
            bias = linear.bias.data
            bias_work = bias.to(device=device, dtype=torch.float64)
            bias_shape = bias_work.shape
            bias_work = torch.matmul(
                bias_work.reshape(-1, bias_shape[-1] // head_dim, head_dim),
                rot,
            ).reshape(bias_shape)
            linear.bias.data = bias_work.to(device=device, dtype=bias.dtype)
    else:
        shape = work.shape
        work = torch.matmul(
            work.reshape(-1, shape[-1] // head_dim, head_dim),
            rot,
        ).reshape(shape)

    linear.weight.data = work.to(device=device, dtype=dtype)


def rotate_model(model, rotate_module, online_rotate, Q=None, r2_by_layer=None, spinquant_ov=False, online_rotate_down=True):
    if rotate_module == 'llm':
        config = model_utils.get_lm_config(model)
        if Q is None:
            Q = get_orthogonal_matrix(config.hidden_size, 'hadamard')
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        model_type = type(model)
        rotate_embeddings(model, Q) # +rotate_projector
        rotate_projectors(model, Q) # +rotate_projector
        rotate_head(model, Q) # is lm head
        utils.cleanup_memory()
        layers = model_utils.get_transformer_layers(model)
        for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
            rotate_attention_inputs(layers[idx], Q)
            rotate_attention_output(layers[idx], Q, model_type)
            rotate_mlp_input(layers[idx], Q, model_type)
            rotate_mlp_output(layers[idx], Q, model_type, online_rotate and online_rotate_down) # no in vit
            if online_rotate:
                R2 = r2_by_layer.get(idx) if r2_by_layer is not None else None
                rotate_ov_proj(
                    layers[idx],
                    model_type,
                    num_heads,
                    head_dim,
                    R2=R2,
                    spinquant_mode=spinquant_ov or bool(r2_by_layer),
                )
        return Q

def get_orthogonal_matrix(size, mode, device=utils.get_dev()):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return hadamard_utils.random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_utils.get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_projectors(model, Q: torch.Tensor) -> None:
    # Rotate the projectors.
    from vlmrot.quant_utils import ActQuantWrapper
    for proj in model_utils.get_projector(model):
        if type(proj) in [nn.Linear, ActQuantWrapper]:
            W = proj # for smolvlm
        elif type(model) == model_utils.LLAVA_HF or type(model) == model_utils.LLAVA_NEXT_HF or type(model) == model_utils.LLAVA_OV_HF:
            W = proj.linear_2
        elif type(model) == QWEN2_VL:
            W = proj.mlp[-1]  # Qwen2-VL's visual_projection_layer is a Linear layer
        else:
            W = proj[-1]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAVA_HF or model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAVA_HF or model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q, model_type, online_rotate, had_dim=-1):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAVA_HF or model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if online_rotate:
        # apply exact (inverse) hadamard on the weights of mlp down proj input
        hadamard_utils.apply_exact_had_to_linear(W, had_dim=had_dim, output=False) 
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_ov_proj(layer, model_type, head_num, head_dim, R2=None, spinquant_mode=False):
    v_proj = layer.self_attn.v_proj
    if (
        model_type == model_utils.LLAVA_HF
        or model_type == model_utils.LLAVA_NEXT_HF
        or model_type == model_utils.LLAVA_OV_HF
        or model_type == QWEN2_VL
        or hasattr(layer.self_attn, "o_proj")
    ):
        o_proj = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    if R2 is not None:
        _apply_head_rotation_to_linear(v_proj, head_dim=head_dim, rotation=R2, output=True)
        _apply_head_rotation_to_linear(o_proj, head_dim=head_dim, rotation=R2, output=False)
        return
    # ??这里Spinquant如果没加载R1，使用的是精确hadamard，而不是随机正交矩阵。
    hadamard_utils.apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    o_had_dim = head_dim if spinquant_mode else -1
    hadamard_utils.apply_exact_had_to_linear(o_proj, had_dim=o_had_dim, output=False)

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert hadamard_utils.is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k

def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    from vlmrot import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)
