import transformers
import torch
from PIL import Image

LLAVA_HF = transformers.models.llava.modeling_llava.LlavaForConditionalGeneration
LLAVA_NEXT_HF = transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration
LLAVA_OV_HF = transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration
QWEN2_VL = transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration

QUESTION_REPLACE_MAPPING = {
    "\nOptions:": "\nChoices:",
    "Please select the correct answer from the options above.": "Answer with the letter.",
}

def get_model(model_name, model_path, device_map='auto'):
    if 'llava-v1.6' in model_name:
        return get_llava_v16(model_path, device_map)
    elif 'llava-onevision' in model_name:
        return get_llava_ov(model_path)
    elif 'Qwen2-VL' in model_name:
        return get_qwen2_vl(model_path, device_map)
    elif 'llava-1.5' in model_name or 'llava-v1.5' in model_name:
        return get_llava_v15(model_path, device_map)
    else:
        raise ValueError(f'Unknown model {model_name}')

def get_vit_config(model):
    model_type = type(model)
    if model_type in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF):
        return get_model_shell(model).vision_tower.config
    elif model_type == QWEN2_VL:
        return model.visual.config
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_lm_config(model):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        return model.config.text_config
    else:
        return model.config

def get_lm_norm_type(model):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF):
        return transformers.models.llama.modeling_llama.LlamaRMSNorm
    elif type(model) == LLAVA_OV_HF:
        return transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
    elif type(model) == QWEN2_VL:
        return transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2RMSNorm
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    
def get_rope_function_name(model):
    if isinstance(model, (LLAVA_HF, LLAVA_NEXT_HF)):
        return "apply_rotary_pos_emb"
    raise NotImplementedError
    
def get_input_masks(model, inputs):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        assert inputs['input_ids'].shape[0] == 1
        image_token_id = getattr(
            model.config,
            "image_token_id",
            getattr(model.config, "image_token_index", None),
        )
        vision_mask = (inputs['input_ids'] == image_token_id)
        caption_mask = (inputs['attention_mask'] == 1) & (~vision_mask)
        return vision_mask, caption_mask
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    
def _clean_question(text):
    text = text.strip()
    for src, dst in QUESTION_REPLACE_MAPPING.items():
        text = text.replace(src, dst)
    return text


def build_answer_only_labels(inputs, question_inputs, tokenizer=None, config=None):
    labels = inputs["input_ids"].clone()
    prompt_len = min(question_inputs["input_ids"].shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100

    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        labels[attention_mask == 0] = -100

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(config, "image_token_index", None)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100

    video_token_id = getattr(config, "video_token_id", None)
    if video_token_id is not None:
        labels[labels == video_token_id] = -100

    return labels


def _message_has_answer(message):
    return any(item.get("type") == "textanw" for item in message)


def build_qwen2vl_train_inputs(message, image_processor, model, device=None):
    from qwen_vl_utils import process_vision_info

    content = []
    answer = ""
    for msg in message:
        if msg["type"] == "image":
            content.append({"type": "image", "image": msg["value"]})
        elif msg["type"] == "text":
            content.append({"type": "text", "text": _clean_question(msg["value"])})
        elif msg["type"] == "textanw":
            answer += msg["value"].strip()

    if not answer:
        raise ValueError("Training sample does not contain a textanw answer field")

    question_conversation = [{"role": "user", "content": content}]
    full_conversation = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    question_prompt = image_processor.apply_chat_template(
        question_conversation,
        add_generation_prompt=True,
    )
    full_prompt = image_processor.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(full_conversation)
    inputs = image_processor(
        text=[full_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    question_inputs = image_processor(
        text=[question_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if device is not None:
        inputs = inputs.to(device, torch.float16)
        question_inputs = question_inputs.to(device, torch.float16)

    inputs["labels"] = build_answer_only_labels(
        inputs,
        question_inputs,
        getattr(image_processor, "tokenizer", None),
        model.config,
    )
    return inputs


def build_llava_next_train_inputs(message, image_processor, model, device=None):
    content = []
    images = []
    answer = ""
    for msg in message:
        if msg["type"] == "image":
            content.append({"type": "image"})
            images.append(Image.open(msg["value"]).convert("RGB"))
        elif msg["type"] == "text":
            content.append({"type": "text", "text": _clean_question(msg["value"])})
        elif msg["type"] == "textanw":
            answer += msg["value"].strip()

    if not answer:
        raise ValueError("Training sample does not contain a textanw answer field")

    question_conversation = [{"role": "user", "content": content}]
    full_conversation = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

    question_prompt = image_processor.apply_chat_template(
        question_conversation,
        add_generation_prompt=True,
    )
    full_prompt = image_processor.apply_chat_template(
        full_conversation,
        add_generation_prompt=False,
    )

    inputs = image_processor(
        text=full_prompt,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    question_inputs = image_processor(
        text=question_prompt,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    if device is not None:
        inputs = inputs.to(device, torch.float16)
        question_inputs = question_inputs.to(device, torch.float16)

    inputs["labels"] = build_answer_only_labels(
        inputs,
        question_inputs,
        getattr(image_processor, "tokenizer", None),
        model.config,
    )
    return inputs


def build_loss_inputs(message, image_processor, model):
    if type(model) == QWEN2_VL and _message_has_answer(message):
        return build_qwen2vl_train_inputs(message, image_processor, model, "cuda")
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF) and _message_has_answer(message):
        return build_llava_next_train_inputs(message, image_processor, model, "cuda")

    inputs = message_to_prompt(message, image_processor, type(model))
    vision_mask, _ = get_input_masks(model, inputs)
    labels = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    labels[vision_mask] = -100
    tokenizer = getattr(image_processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    inputs["labels"] = labels
    return inputs
    
def get_embeddings(model):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        return [get_language_model(model).embed_tokens]
    else:
        raise ValueError(f'Unknown model type {type(model)}')


def get_model_shell(model):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF) and hasattr(model, "model"):
        return model.model
    return model


def get_language_model(model):
    if type(model) == LLAVA_HF:
        language_model = get_model_shell(model).language_model
        # Older Transformers wraps the decoder and lm_head in LlamaForCausalLM.
        return getattr(language_model, "model", language_model)
    if hasattr(model, "language_model"):
        return model.language_model
    shell = get_model_shell(model)
    if hasattr(shell, "language_model"):
        return shell.language_model
    raise ValueError(f"Cannot find language_model on {type(model)}")

def get_projector(model) -> list[torch.nn.Module]:
    model_type = type(model)
    if model_type in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF):
        return [get_model_shell(model).multi_modal_projector]
    elif model_type == QWEN2_VL:
        return [model.visual.merger]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_vit_layers(model):
    model_type = type(model)
    if model_type in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF):
        return [layer for layer in get_model_shell(model).vision_tower.vision_model.encoder.layers]
    elif model_type == QWEN2_VL:
        return [layer for layer in model.visual.blocks]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_transformer_layers(model):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        return [layer for layer in get_language_model(model).layers]
    else:
        raise ValueError(f'Unknown model type {type(model)}')
    
def get_layers(model):
    if type(model) in (LLAVA_HF, LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        return get_language_model(model).layers
    raise NotImplementedError
    
def get_pre_head_layernorm(model):
    model_type = type(model)
    pre_head_layernorm = get_language_model(model).norm
    norm_type = get_lm_norm_type(model)
    assert isinstance(pre_head_layernorm, norm_type)
    return pre_head_layernorm

def get_lm_head(model):
    if type(model) == LLAVA_HF:
        return model.get_output_embeddings()
    elif type(model) in (LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {type(model)}')


def set_lm_head(model, module):
    if type(model) == LLAVA_HF:
        model.set_output_embeddings(module)
        return
    elif type(model) in (LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL):
        model.lm_head = module
        return
    raise ValueError(f'Unknown model type {type(model)}')

def get_llava_v16(model_path, device_map='auto'):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        # Use the same low-memory attention backend for the original target,
        # FlatQuant calibration and final quantized inference.
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=device_map
    )
    model.seqlen = 2048
    processor = LlavaNextProcessor.from_pretrained(model_path)
    return model, processor

def get_llava_ov(model_path):
    from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model.seqlen = 2048
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    return model, processor

def get_qwen2_vl(model_path, device_map):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map
    )
    model.seqlen = 2048
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def get_llava_v15(model_path, device_map='auto'):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.seqlen = 2048
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def message_to_prompt(message, image_processor, model_type):
    if model_type in (LLAVA_HF, LLAVA_NEXT_HF):
        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                content.append({"type": "image"})
                images.append(Image.open(msg['value']).convert('RGB'))
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        conversation = [{"role": "user", "content": content}]
        prompt = image_processor.apply_chat_template(conversation,
                                                    add_generation_prompt=True)
        inputs = image_processor(text=prompt, images=images, return_tensors='pt').to('cuda', torch.float16)
        return inputs

    elif model_type == LLAVA_OV_HF:
        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                content.append({"type": "image", "image": None})
                images.append(Image.open(msg['value']).convert('RGB'))
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        conversation = [{"role": "user", "content": content}]
        prompt = image_processor.apply_chat_template(conversation,
                                                     add_generation_prompt=True)
        inputs = image_processor(text=prompt, images=images, return_tensors='pt')
        inputs = inputs.to('cuda', torch.float16)
        return inputs

    elif model_type == QWEN2_VL:
        from qwen_vl_utils import process_vision_info
        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                content.append({"type": "image", "image": msg['value']})
            elif msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
        conversation = [{"role": "user", "content": content}]
        prompt = image_processor.apply_chat_template(conversation,
                                                    add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        assert video_inputs is None
        inputs = image_processor(text=[prompt], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors='pt')
        inputs = inputs.to('cuda', torch.float16)
        return inputs

    else:
        raise NotImplementedError

class LN_(torch.nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.ln = ori_layer_norm

    def forward(self, x):
        out = self.ln(x)
        outdtype = out.dtype
        out_ = out.double()
        out_ = out_ - out_.mean(dim=-2, keepdim=True)
        return out_.to(outdtype)
    
class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)

# class RMSLN(torch.nn.Module):
#     """
#     This class implements the Root Mean Square Normalization (RMSN) layer.
#     We use the implementation from LLAMARMSNorm here:
#     https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
#     """

#     def __init__(self, mean_dim: int, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.mean_dim = mean_dim
#         self.weight = torch.nn.Parameter(torch.ones(1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         input_dtype = x.dtype
#         if x.dtype == torch.float16:
#             x = x.to(torch.float32)
#         variance = x.pow(2).mean(-1, keepdim=True)
#         x = x * torch.rsqrt(variance + self.eps)
#         return x.to(input_dtype)
    
def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)
