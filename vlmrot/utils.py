import argparse, os, yaml, logging, pprint, random, gc
import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_submodules
import numpy as np
from vlmrot import model_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

supported_models = [
            'llava-hf/llava-v1.6-vicuna-7b-hf',
            'llava-hf/llava-v1.6-vicuna-13b-hf',
            'llava-hf/llava-onevision-qwen2-7b-ov-hf',
            'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2-VL-2B-Instruct',
            'llava-hf/llava-1.5-7b-hf',
            'llava-hf/llava-1.5-13b-hf',
            ]

supported_cal_datasets = [
    'ScienceQA_Train'
]

supported_eval_datasets = [
    'ScienceQA_TEST',
    'MMMU_DEV_VAL',
    'OCRBench',
    'VizWiz',
    'TextVQA_VAL',
    'ChartQA_TEST',
    'SEEDBench2_Plus',
]
    # # For lmms-eval
    # 'mmmu_val', 
    # 'seedbench', 
    # 'ocrbench', 
    # 'vizwiz_vqa_val', 
    # 'scienceqa', 
    # 'textvqa_val', 
    # 'docvqa_val', 
    # 'seedbench_2_plus', 
    # 'chartqa', 
    # 'mme', 
    # 'mme_realworld', 
    # 'mme_realworld_cn'

# Unified parameter choices configuration (single source of truth)
PARAM_CHOICES = {
    'run_mode': ['single_card', 'data_parallel', 'model_distribute'],
    'model': supported_models,
    'quant_method': ['rtn', 'gptq'],
    'cal_dataset': supported_cal_datasets,
    'eval_dataset': supported_eval_datasets
}

# Parameters that accept multiple values and need list conversion
MULTI_VALUE_PARAMS = {'eval_dataset'}

def parser_gen():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--run_mode', type=str, default='single_card', help='GPU situations decide the run mode',
                         choices=PARAM_CHOICES['run_mode'])
    parser.add_argument('--config', default=None, type=str, help='Path to the config file.')

    # model
    parser.add_argument('--model', type=str, default='liuhaotian/llava-v1.5-7b',
                        help='Model to load', choices=PARAM_CHOICES['model'])
    parser.add_argument('--model_path', type=str, default=None, help='Path to the selected model weights.')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')

    # calibration
    parser.add_argument('--mbq', action=argparse.BooleanOptionalAction, default=False,
                        help='Calibrate and evaluate modality-balanced weight/activation quantization.')
    parser.add_argument('--mbq_scale_path', type=str, default=None,
                        help='Optional MBQ scale cache file.')
    parser.add_argument('--mbq_load_scales', action=argparse.BooleanOptionalAction, default=False,
                        help='Load MBQ scales instead of calibrating.')
    parser.add_argument('--mbq_reweight', action=argparse.BooleanOptionalAction, default=True,
                        help='Use visual/text gradient ratios in MBQ scale search.')
    parser.add_argument('--mbq_loss', choices=('mae', 'mse'), default='mae')
    parser.add_argument('--mbq_offload_activations', action=argparse.BooleanOptionalAction, default=True,
                        help='Save MBQ backward tensors on CPU to reduce GPU memory during gradient collection.')
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--quant_vit', type=bool, default=True, help='Quant the vit encoder or not')
    parser.add_argument('--scale_smooth', type=bool, default=True, help='Smoothquant-like scale smooth')
    parser.add_argument('--smoothquant', type=bool, default=False, help='Use smoothquant to scale or not')
    parser.add_argument('--quarot', type=bool, default=False, help='Use Quarot method to rotate')
    parser.add_argument('--online_rotate_down', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable paired down_proj Hadamard when QuaRot online rotation is enabled')
    parser.add_argument('--online_rotate', type=bool, default=False, help='Add online hadmard for ov, and down_proj linear')
    parser.add_argument('--spinquant', type=bool, default=False, help='Use Spinquant method to rotate.')
    parser.add_argument('--trained_R1_R2_path', type=str, default=None,
                        help='Path to a SpinQuant rotation checkpoint containing R1 and optional per-layer R2')
    parser.add_argument('--flatquant', action=argparse.BooleanOptionalAction, default=False,
                        help='Use FlatQuant adapters and load trained FlatQuant matrices for evaluation.')
    parser.add_argument('--flatquant_matrix_path', type=str, default=None,
                        help='Path to flat_matrices.pth or a directory containing it.')
    parser.add_argument('--subspace', action=argparse.BooleanOptionalAction, default=False,
                        help='Load trained subspace FlatQuant transformations for evaluation.')
    parser.add_argument('--subspace_matrix_path', type=str, default=None,
                        help='Path to subspace_matrices.pth or a directory containing it.')
    parser.add_argument('--subspace_cache_dir', type=str, default='cache/subsapce',
                        help='Directory containing the fixed per-layer U_modal bases.')
    parser.add_argument('--subspace_record_path', type=str, default='record.txt',
                        help='Grouping record used while training subspace matrices.')
    parser.add_argument('--save_flatquant_dir', type=str, default='cache/flatquant/',
                        help='Directory for FlatQuant training outputs.')
    parser.add_argument('--flatquant_modality_grad_path', type=str, default=None,
                        help='Path to cached per-layer visual/text gradient weights for FlatQuant MBQ-MAE training.')
    parser.add_argument('--flat_lr', type=float, default=5e-3,
                        help='Learning rate for FlatQuant transformation calibration.')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of FlatQuant calibration epochs.')
    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='Gradient accumulation interval for FlatQuant calibration.')
    parser.add_argument('--cali_trans', action=argparse.BooleanOptionalAction, default=False,
                        help='Calibrate FlatQuant transformation matrices.')
    parser.add_argument('--add_diag', action=argparse.BooleanOptionalAction, default=False,
                        help='Add FlatQuant per-channel diagonal scaling.')
    parser.add_argument('--lwc', action=argparse.BooleanOptionalAction, default=False,
                        help='Use FlatQuant learnable weight clipping.')
    parser.add_argument('--lac', action=argparse.BooleanOptionalAction, default=False,
                        help='Use FlatQuant learnable activation clipping.')
    parser.add_argument('--diag_init', type=str, default='sq_style',
                        choices=('sq_style', 'one_style'),
                        help='FlatQuant diagonal scale initialization.')
    parser.add_argument('--diag_alpha', type=float, default=0.3,
                        help='SmoothQuant-style alpha for FlatQuant diagonal initialization.')
    parser.add_argument('--direct_inv', action=argparse.BooleanOptionalAction, default=False,
                        help='Use direct inverse FlatQuant transforms instead of SVD parameterization.')
    parser.add_argument('--separate_vtrans', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep FlatQuant V-cache transform separate from v_proj reparameterization.')

    parser.add_argument('--split_rotate', action=argparse.BooleanOptionalAction, default=False,
                        help='Split OV rotation channels by gradient importance and fuse the split rotation into weights')

    parser.add_argument('--parrot', type=bool, default=False)
    parser.add_argument('--parrot_mode', type=str, default='activation', choices=('activation', 'gradient'))
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--k_bits', type=int, default=16, help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--v_bits', type=int, default=16, help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--q_bits', type=int, default=16,
                        help='Number of bits for query quantization in FlatQuant-style cache calibration.')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--q_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric query quantization (default: False)')
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric K-cache quantization (default: False)')
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='Asymmetric V-cache quantization (default: False)')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=True,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--k_groupsize', type=int, default=-1, 
                        help='Groupsize for key quantization.')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for key quantization. new_max = max * clip_ratio')
    parser.add_argument('--lma_clip_ratio', type=float, default=1.0,
                        help='override Clip ratio for lm activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for V-cache/output activation quantization.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default=['MMMU_val'], 
                        help='Calibration dataset', choices=PARAM_CHOICES['cal_dataset'])
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')
    parser.add_argument('--question_file', type=str, default=None, 
                        help="Path to calibration dataset question file(ScienceQA)")
    parser.add_argument('--image_base', type=str, default=None, 
                        help="Path to calibration dataset image root(ScienceQA)")

    # quantization
    parser.add_argument('--quant_method', type=str, default='rtn',
                        help='Method when quantize the model weights', choices=PARAM_CHOICES['quant_method'])
    parser.add_argument('--load_qmodel', type=bool, default=False, help='Load the quantized model')
    parser.add_argument('--save_qmodel_dir', type=str, default=None, 
                        help='Path to save the quantized model, None means do not save')
    parser.add_argument('--load_backward_result', type=bool, default=False, help='Load the gard statistic in backward')
    parser.add_argument('--save_backward_result_dir', type=str, default=None, help='Dir to the grad statistic')
    parser.add_argument('--load_smooth_result', type=bool, default=False, help="load the result of scale smooth")
    parser.add_argument('--save_smooth_result_dir', type=str, default=None, help='Dir to result of scale smooth')
    # evaluation
    parser.add_argument('--eval_dataset', type=str, default=['MMMU_val'],
                        help='evaluation tasks', nargs="+", choices=PARAM_CHOICES['eval_dataset'])
    parser.add_argument('--eval_log_path', type=str, default='./log/', help='Path to save the output eval results log.')
    parser.add_argument('--eval_result_dir', type=str, default='./cache/eval_results', help="Path to save eval results")

    # speed test

    # profile
    parser.add_argument('--profile', type=bool, default=False, help='profile the quantize process')
    parser.add_argument('--profile_dir', type=str, default=None, help='Save dir of the profile output')
    parser.add_argument('--error_check', action=argparse.BooleanOptionalAction, default=False,
                        help='Compare the original model and the current model on identical inputs.')
    parser.add_argument('--error_check_samples', type=int, default=10,
                        help='Number of samples used for original/current model output error checking.')
    parser.add_argument('--error_check_topk', type=int, default=None,
                        help='Absolute number of top important channels to track by gradient magnitude.')
    parser.add_argument('--error_check_topk_percent', type=float, default=90.0,
                        help='Percentage of channels to consider important (default: 10%)')
    parser.add_argument('--error_check_plot_mode', type=str, default='channel_diagnostics',
                        choices=('relative_error_histogram', 'channel_diagnostics', 'both'),
                        help='Plot mode for layer input error_check results.')
    parser.add_argument('--split_error_sweep', action=argparse.BooleanOptionalAction, default=False,
                        help='Run error_check sweep over online_channel_split_topk_percent values.')

    args = parser.parse_args()

    # load config from file
    if args.config:
        assert os.path.exists(args.config)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                setattr(args, k, v)

    # print config
    # logging.info('Arguments: ')
    # logging.info(pprint.pformat(vars(args)))
    # logging.info('--' * 30)

    set_seed(args.seed)

    return args


def validate_args(args):
    """
    Validate all argument values are within allowed choices.

    This function:
    - Validates all choice-restricted parameters against PARAM_CHOICES
    - Normalizes multi-value parameters to lists
    - Raises ValueError with clear error messages if validation fails

    Args:
        args: argparse.Namespace object

    Raises:
        ValueError: If any parameter value is not in the allowed choices
    """
    errors = []
    if getattr(args, 'mbq', False):
        from vlmrot.mbq_utils import validate_mbq_args
        validate_mbq_args(args)
    input_paths = {
        'model_path': args.model_path,
        'question_file': args.question_file,
        'image_base': args.image_base
    }
    if getattr(args, 'mbq', False):
        # Cached MBQ evaluation and VLMEval datasets need no ScienceQA files.
        input_paths = {'model_path': args.model_path}
        dataset = args.cal_dataset
        dataset = dataset[0] if isinstance(dataset, (list, tuple)) else dataset
        if not (args.load_qmodel or args.mbq_load_scales) and 'scienceqa' in dataset.lower():
            input_paths.update(question_file=args.question_file, image_base=args.image_base)
    if args.flatquant:
        input_paths['flatquant_matrix_path'] = args.flatquant_matrix_path
        if args.flatquant_matrix_path and os.path.basename(
            os.path.dirname(os.path.normpath(args.flatquant_matrix_path))
        ) != args.tag:
            errors.append('args.flatquant_matrix_path not match args.tag!')
    if args.subspace:
        subspace_matrix_path = (
            args.subspace_matrix_path or args.flatquant_matrix_path
        )
        input_paths['subspace_matrix_path'] = subspace_matrix_path
        input_paths['subspace_cache_dir'] = args.subspace_cache_dir
    for k, v in input_paths.items():
        if not v or not os.path.exists(v):
            errors.append(f'args.{k}: {v} not exist!')
    if sum([args.quarot, args.spinquant, args.flatquant, args.subspace]) > 1:
        errors.append(
            'quarot, spinquant, flatquant and subspace cannot be implemented together'
        )
    
    if errors:
        error_msg = "Argument validation failed:\n- " + "\n- ".join(errors)
        logging.error(error_msg)
        raise ValueError(error_msg)
    
def set_seed(seed):
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python's built-in random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU & CUDA
    torch.cuda.manual_seed(seed)  # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimizations for determinism

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def get_dev():
    return (
        torch.device(f'cuda:{dist.get_rank()}') if dist.is_initialized() else 
        torch.device('cuda') if torch.cuda.is_available() else 
        torch.device('cpu'))

def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

def dispatch_model_to_cuda(model):
    device_map = model.hf_device_map
    model = dispatch_model(model, device_map=device_map)
    if type(model) == model_utils.LLAVA_NEXT_HF:
        model.model.image_newline.data = model.model.image_newline.data.to('cuda')
    # for name, param in model.named_parameters():
    #     print(name, param.device, param.dtype, tuple(param.shape))
    return model

def move_model_to_cpu(model):
    remove_hook_from_submodules(model)
    model = model.cpu()
    return model

@torch.no_grad()
def get_free_gpu():
    free_mem = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_mem.append(free)
    best_gpu = int(torch.argmax(torch.tensor(free_mem)))
    return best_gpu, free_mem

def list_tensors_on_gpu(device_id=0):
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.device.index == device_id:
                tensors.append(obj)
        except:
            pass
    return tensors

def move_list_tensor(x: list, device):
    for i in range(len(x)):
        old = x[i]
        x[i] = old.detach().to(device)
        del old
    torch.cuda.empty_cache()

def del_list_tensor(x: list):
    for tensor in x:
        del tensor
    torch.cuda.empty_cache()

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size
