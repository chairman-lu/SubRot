import torch
from vlmeval.smp import *
from vlmeval.smp import dump, get_rank_and_world_size, load
import os, logging, string
from tqdm import tqdm
from vlmrot import model_utils, utils
import pandas as pd
from tabulate import tabulate

def is_vizwiz_dataset(eval_dataset):
    return str(eval_dataset).lower() == 'vizwiz'

def build_prompt_llava(line, dataset, eval_dataset):
    tgt_path = dataset.dump_image(line)
    question = line['question']
    hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
    if hint is not None:
        question = hint + "\n" + question 
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    if is_vizwiz_dataset(eval_dataset):
        question += "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."
    elif eval_dataset in {"TextVQA_VAL", "ChartQA_TEST"}:
        question += "\nAnswer the question using a single word or phrase."
    elif eval_dataset == "OCRBench":
        question += "\nPlease try to answer the question with short words or phrases if possible."
    else:
        for key, item in options.items():
            question += f"\n{key}. {item}"
        question += "\nAnswer with the option's letter from the given choices directly."
    message = [dict(type="image", value=s) for s in tgt_path]
    message.append(dict(type="text", value=question))
    return message

def output_process(answer):
    if "<s>" in answer:
        answer = answer.replace("<s>", "").strip()
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[1].strip()
    elif "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[1].strip()
    elif "assistant\n" in answer:
        answer = answer.split("assistant\n")[1].strip()
    elif "<|end_header_id|>\n\n" in answer:
        answer = answer.split("<|end_header_id|>\n\n")[2].strip()

    if "</s>" in answer:
        answer = answer.split("</s>")[0].strip()
    elif "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()
    elif "<|eot_id|>" in answer:
        answer = answer.split("<|eot_id|>")[0].strip()
    return answer

def infer_data_llava(model, image_processor, dataset, args, eval_dataset):
    sheet_indices = range(len(dataset))
    data = dataset.data.iloc[sheet_indices]
    length1 = len(data)
    out_dir = os.path.join(args.eval_result_dir, 
                           args.model.split('/')[-1] + 
                           f'_W{args.w_bits}A{args.a_bits}K{args.k_bits}V{args.v_bits}' + 
                           ('_Smoothquant' if args.smoothquant else '') +
                           ('_scale' if args.scale_smooth else '') +
                        #    ('_fullRotate' if (args.offline_rotate and args.online_rotate) else '') +
                        #    ('_offlineRotate' if (args.offline_rotate and not args.online_rotate and not args.split_rotate) else '') +
                        #    ('_splitOV' if (args.offline_rotate and args.split_rotate) else '') +
                        #    ('_splitR1' if (args.offline_rotate and args.split_r1_rotate) else '') +
                           ('_Quarot' if args.quarot else '') +
                           ('_SpinQuant' if args.spinquant else '') +
                           ('_FlatQuant' if args.flatquant else '') +
                           ('_Subspace' if args.subspace else '') +
                           ('_rtn' if (args.quant_method == 'rtn' and (args.a_bits < 16 or args.v_bits < 16)) else '_gptq') +
                           (f'_{args.tag}'),
                           eval_dataset)
    logging.info(f'eval out dir: {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, f'vlm_eval.pkl')
    data_indices = [i for i in data['index']]
    res = {}
    if os.path.exists(out_file):
        res.update(load(out_file))
    data = data[~data['index'].isin(res)]   # resume, avoiding recompute the results
    length2 = len(data)
    if length1 > length2:
        logging.info(f'Resumed! Original length of {eval_dataset}: {length1}, current: {length2}')
    else:
        logging.info(f'{length1} samples in {eval_dataset}')
    for i in tqdm(range(len(data))):
        idx = data.iloc[i]['index']
        message = build_prompt_llava(data.iloc[i], dataset, eval_dataset)
        inputs = model_utils.message_to_prompt(
        message, image_processor, type(model)
        )
        if type(model) == model_utils.LLAVA_NEXT_HF or type(model) == model_utils.QWEN2_VL or type(model) == model_utils.LLAVA_HF:
            if type(model) == model_utils.QWEN2_VL:
                model.generation_config.temperature = None
                model.generation_config.top_k = None
                model.generation_config.top_p = None
            output = model.generate(
                **inputs,
                do_sample=False, 
                max_new_tokens=16,
                top_p=None,
                num_beams=1
            )
            # new_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
            # logging.info(f"new_tokens={new_tokens}")
        elif type(model) == model_utils.LLAVA_OV_HF:
            output = model.generate(
                **inputs,
                do_sample=False, 
                max_new_tokens=16,
                top_p=None,
                num_beams=1,
                eos_token_id=image_processor.tokenizer.pad_token_id,
                pad_token_id=image_processor.tokenizer.pad_token_id
            )
        response = image_processor.decode(output[0], skip_special_tokens=True)
        response = output_process(response) 
        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)
    dump({k: res[k] for k in data_indices}, out_file)
    return out_file

@torch.no_grad()
def evaluator(model, dataset, device, args, image_processor, eval_dataset):
    model = utils.dispatch_model_to_cuda(model)
    # model = model.cuda()
    if not os.path.exists(args.eval_result_dir):
        os.mkdir(args.eval_result_dir)
    out_file = infer_data_llava(model, image_processor, dataset, args, eval_dataset)
    result_file = os.path.join(args.eval_result_dir, 'vlm_result.xlsx')
    if os.path.exists(result_file):
        for file in os.listdir(args.eval_result_dir):
            if file.split('_')[0] == 'vlm':
                remove_file = os.path.join(args.eval_result_dir, file)
                os.remove(remove_file)
                logging.info(f'removed previous file: {remove_file}')
    data_all = load(out_file)
    data = dataset.data

    for x in data['index']:
        assert x in data_all
    data['prediction'] = [str(data_all[x]) for x in data['index']]
    if 'image' in data:
        data.pop('image')
    dump(data, result_file)

    judge_kwargs = {'nproc': 4, 'verbose': True, 'retry': 3}
    eval_results = dataset.evaluate(result_file, **judge_kwargs)

    import json
    logging.info(eval_results)
    if isinstance(eval_results, dict):
        logging.info('\n' + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        if len(eval_results) < len(eval_results.columns):
            eval_results = eval_results.T
        logging.info('\n' + tabulate(eval_results))
    model = utils.move_model_to_cpu(model)
    utils.cleanup_memory()
    return eval_results
