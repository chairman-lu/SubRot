import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd.graph import save_on_cpu
import argparse, logging, functools, os, copy, datetime
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from vlmrot import utils, model_utils, data_utils, rotation_utils, smooth_utils, parrot_utils, \
                   quant_utils, hadamard_utils, gptq_utils, eval_utils, vis_utils, check_utils, rotation_split_utils, \
                   flatquant_utils

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


def _matches_target_layer(name, target_layer_idx):
    return target_layer_idx is None or f"layers.{target_layer_idx}." in name


def _apply_weight_quantization(
    model, args, image_processor=None, save_qmodel=True, target_layer_idx=None
):
    assert args.w_bits < 16
    save_dict = {}
    if args.load_qmodel and args.save_qmodel_dir:
        rotate = '_rotate' if args.offline_rotate else ''
        save_path = os.path.join(args.save_qmodel_dir, f"{args.model.split('/')[-1]}_W{args.w_bits}A{args.a_bits}{rotate}.pt")
        assert os.path.exists(save_path)
        logging.info(f'Loading quantized model from {save_path}')
        save_dict = torch.load(save_path, weights_only=False)
        model.load_state_dict(save_dict['model'])
    elif args.quant_method == 'rtn':
        original_w_clip = getattr(args, "w_clip", False)
        if (getattr(args, "flatquant", False) or getattr(args, "subspace", False)) and original_w_clip:
            logging.info(
                "FlatQuant evaluation: disabling RTN MSE weight clipping to match FlatQuant training fake quantizer."
            )
            args.w_clip = False
        try:
            quantizers = gptq_utils.rtn_fwrd(
                model,
                utils.get_dev(),
                args,
                target_layer_idx=target_layer_idx,
            )
        finally:
            args.w_clip = original_w_clip
        save_dict["w_quantizers"] = quantizers
    elif args.quant_method == 'gptq':
        if image_processor is None:
            raise ValueError('VLM GPTQ requires the model image processor')
        dataloader = data_utils.get_loaders(
            args.cal_dataset, args.nsamples, args.seed,
            model.seqlen, args.question_file, args.image_base
        )
        quantizers = gptq_utils.gptq_fwrd(
            model, dataloader, utils.get_dev(), args, image_processor=image_processor
        )
        save_dict["w_quantizers"] = quantizers

    if save_qmodel and args.save_qmodel_dir:
        if not os.path.exists(args.save_qmodel_dir):
            os.makedirs(args.save_qmodel_dir)
        save_dict["model"] = model.state_dict()
        rotate = '_rotate' if args.offline_rotate else ''
        save_path = os.path.join(args.save_qmodel_dir, f"{args.model.split('/')[-1]}_W{args.w_bits}A{args.a_bits}{rotate}.pt")
        torch.save(save_dict, save_path)
        logging.info(f'Quantized model saved at: {save_path}')

    return save_dict


def _configure_activation_quantization(model, args, target_layer_idx=None):
    assert args.a_bits < 16 or args.v_bits < 16
    if target_layer_idx is None:
        logging.info(f'setting a clip ratio in lm {min(args.a_clip_ratio, args.lma_clip_ratio)}')
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    if not qlayers:
        raise RuntimeError(
            'Activation quantization requested, but no ActQuantWrapper was installed.'
        )
    logging.info('Configuring activation quantization: %d wrappers, A%d, V%d',
                 len(qlayers), args.a_bits, args.v_bits)
    down_proj_groupsize = -1
    if args.a_groupsize > 0 and "llama" in args.model:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
    for name in qlayers:
        if not _matches_target_layer(name, target_layer_idx):
            continue
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not(args.a_asym)
        layer_a_clip = min(args.a_clip_ratio, args.lma_clip_ratio)

        if args.quarot:
            v_groupsize = -1
        elif args.spinquant or args.split_rotate:
            num_heads = model_utils.get_lm_config(model).num_attention_heads
            model_dim = model_utils.get_lm_config(model).hidden_size
            head_dim = model_dim // num_heads
            v_groupsize = head_dim
            if "o_proj" in name:
                layer_groupsize = head_dim

        if 'v_proj' in name and args.v_bits < 16:
            qlayers[name].out_quantizer.configure(
                bits=args.v_bits,
                groupsize=v_groupsize,
                sym=not(args.v_asym),
                clip_ratio=args.v_clip_ratio
            )

        if 'lm_head' in name:
            layer_input_bits = 16

        if 'down_proj' in name:
            if args.int8_down_proj:
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(
            bits=layer_input_bits,
            groupsize=layer_groupsize,
            sym=layer_a_sym,
            clip_ratio=layer_a_clip,
        )

def _get_text_config(model):
    config = getattr(model, "config", None)
    return getattr(config, "text_config", config)


def _configure_online_rotation(model, args, has_spinquant_r2=False, online_rotate_down=True):
    qlayers = quant_utils.find_qlayers(model)
    text_config = _get_text_config(model)

    for name in qlayers:
        if 'down_proj' in name and online_rotate_down:
            had_K, K = hadamard_utils.get_hadK(text_config.intermediate_size)
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].fp32_had = args.fp32_had
        if 'o_proj' in name:
            if has_spinquant_r2:
                qlayers[name].online_partial_had = False
                continue
            had_K, K = hadamard_utils.get_hadK(text_config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = text_config.hidden_size // text_config.num_attention_heads
            qlayers[name].fp32_had = args.fp32_had


if __name__ == '__main__':
    args = utils.parser_gen()
    utils.validate_args(args)
    model, image_processor = model_utils.get_model(args.model, args.model_path)
    model.eval()
    model = utils.move_model_to_cpu(model)
    utils.cleanup_memory(verbos=False)

    if args.error_check:
        logging.info('[error_check] keeping a copy of the original model before calibration')
        original_model = copy.deepcopy(model).eval()
        error_check_inputs = check_utils._prepare_error_check_inputs(
            args, image_processor, model, cached_inputs=None
        )

    # Plain LLM RTN/GPTQ path: install wrappers before weight quantization.
    # Their default is A16; activation quantization is enabled after calibration.
    if (not args.scale_smooth) and (not args.quarot) and (not args.spinquant) and (not args.flatquant) \
        and (not args.subspace) and (not args.mbq) \
        and (not args.smoothquant) and args.quant_method in ('rtn', 'gptq'):
        if args.a_bits < 16 or args.v_bits < 16:
            logging.info('plain %s: adding activation quantization wrappers to LM layers', args.quant_method.upper())
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            if not qlayers:
                quant_utils.add_actquant(model_utils.get_layers(model))
                
    # Plain LLM SmoothQuant path: SmoothQuant without rotation.
    if args.smoothquant:
        assert not args.scale_smooth
        assert not args.offline_rotate
        if args.load_smooth_result and args.save_smooth_result_dir:
            load_path = os.path.join(args.save_smooth_result_dir,
                                     args.model.split('/')[-1], 'smoothquant_scales.pt')
            smooth_results = torch.load(load_path, weights_only=False)
            logging.info(f'loaded SmoothQuant act scales from {load_path}')
        else:
            dataloader = data_utils.get_loaders(
                args.cal_dataset, args.nsamples, args.seed,
                model.seqlen, args.question_file, args.image_base
            )
            utils.set_seed(args.seed)
            smooth_results = smooth_utils.get_smoothquant_act_scales(
                dataloader, model, image_processor, args
            )
        if args.save_smooth_result_dir and not args.load_smooth_result:
            save_dir = os.path.join(args.save_smooth_result_dir, args.model.split('/')[-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'smoothquant_scales.pt')
            torch.save(smooth_results, save_path)
            logging.info(f'SmoothQuant act scales saved at {save_path}')
        logging.info('plain SmoothQuant: applying scales to LM layers')
        smooth_utils.smooth_lm(model.language_model, smooth_results)
        if args.a_bits < 16 or args.v_bits < 16:
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            if not qlayers:
                quant_utils.add_actquant(model_utils.get_layers(model))           

    if args.mbq:
        from vlmrot.mbq_utils import run_mbq
        run_mbq(model, image_processor, args)
    # llm activation rotation and quantization
    Q = None
    assert sum([args.quarot, args.spinquant, args.split_rotate, args.flatquant, args.subspace]) <= 1
    if args.quarot:
        # assert args.online_rotate == True
        logging.info(f'Using Quarot, online rotate is {args.online_rotate}')
        rotation_utils.fuse_layernorm(model, 'llm')
        Q = rotation_utils.rotate_model(
            model, 'llm', args.online_rotate, online_rotate_down=args.online_rotate_down
        )
        utils.cleanup_memory()
        quant_utils.add_actquant(model_utils.get_layers(model))
        if args.online_rotate:
            _configure_online_rotation(model, args, online_rotate_down=args.online_rotate_down)
    elif args.spinquant:
        if args.trained_R1_R2_path:
            R1, R2_by_layer = rotation_utils.load_optimized_R1_R2(args.trained_R1_R2_path, device=utils.get_dev())
            logging.info(f'Loaded SpinQuant trained R1 from {args.trained_R1_R2_path} (optional R2 layers: {len(R2_by_layer)})')
        else:
            R1, R2_by_layer = None, None
            logging.info(f'Using Spinquant without loading R1 R2.')
        rotation_utils.fuse_layernorm(model, 'llm')
        Q = rotation_utils.rotate_model(model, 'llm', True, R1, R2_by_layer, spinquant_ov=True)
        utils.cleanup_memory()
        quant_utils.add_actquant(model_utils.get_layers(model))
        _configure_online_rotation(model, args, has_spinquant_r2=True)
    elif args.flatquant:
        logging.info(f'Using FlatQuant matrices from {args.flatquant_matrix_path}')
        flatquant_utils.apply_flatquant_to_model(args, model)
        flatquant_utils.load_flat_matrices(args, model, args.flatquant_matrix_path)
        flatquant_utils.reparameterize_model(model)
        utils.cleanup_memory()
    # elif args.subspace:
    #     logging.info(
    #         "Using subspace matrices from %s",
    #         getattr(args, "subspace_matrix_path", None)
    #         or getattr(args, "flatquant_matrix_path", None),
    #     )
    #     flatquant_utils.apply_flatquant_to_model(args, model)
    #     subspace_utils.load_subspace_matrices(
    #         args, model, prepare_for_evaluation=True
    #     )
    #     utils.cleanup_memory()

    if args.error_check:
        logging.info('[error_check] comparing original model with current calibration model')
        # check_utils._log_model_output_error(original_model, model, error_check_inputs)
        check_utils._compute_and_log_channel_relative_error(
            original_model, model, error_check_inputs, args,
            phase='after_rotation', rotation_matrix=Q
        )
        utils.cleanup_memory(verbos=False)

    # llm weight quantization
    if args.w_bits < 16 and not args.mbq:
        _apply_weight_quantization(
            model, args, image_processor=image_processor, save_qmodel=True
        )

    # Activation quantization configs
    if (args.a_bits < 16 or args.v_bits < 16) and (not args.flatquant) and (not args.subspace) and (not args.mbq):
        _configure_activation_quantization(model, args)
        
    if args.k_bits < 16 and (not args.flatquant) and (not args.subspace):
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                    "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        layer.self_attn, 
                        rope_function_name, 
                        config=model.config.text_config,
                        **k_quant_config)
        logging.info('finish qk rotation and k quant')
            
    if args.error_check:
        logging.info('[error_check] comparing original model with current quantization model')
        check_utils._log_model_output_error(original_model, model, error_check_inputs)
        check_utils._compute_and_log_channel_relative_error(
            original_model, model, error_check_inputs, args,
            phase='after_quantization', rotation_matrix=Q
        )
        del original_model, error_check_inputs
        utils.cleanup_memory(verbos=False)
        exit()

    for eval_dataset in args.eval_dataset:
        test_loader = data_utils.get_loaders(eval_dataset)
        dataset_ppl = eval_utils.evaluator(model, test_loader, utils.get_dev(), args, image_processor, eval_dataset)

