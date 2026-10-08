
import json, os, math, random

def get_loaders(name, nsamples=128, seed=42, seqlen=2048, question_file=None, image_base=None):
    if 'scienceqa' in name.lower():
        return get_vlm_data(name, nsamples, seed, question_file, image_base)
    else:
        from vlmeval.dataset import build_dataset
        return build_dataset(name)
    
def get_vlm_data(name, nsamples, seed, question_file, image_base):
    if 'train' in name.lower():
        def build_prompt_llava_train(line, pathbase):
            question = line['conversations'][0]
            question = question['value'].replace('<image>', '').strip()
            tgt_path = [os.path.join(pathbase, line["image"])]
            anwser = line['conversations'][1]['value'].strip()
            message = [dict(type="image", value=s) for s in tgt_path]
            message.append(dict(type="text", value=question))
            message.append(dict(type="textanw", value=anwser))
            return message
        
        dataloader = []
        dataset = json.load(open(os.path.expanduser(question_file), "r"))
        random.seed(seed)
        indices = random.sample(range(len(dataset)), nsamples)
        for i in indices:
            line = dataset[i]
            dataloader.append(build_prompt_llava_train(line, image_base))
        return dataloader

    else:
        from vlmeval.dataset import build_dataset
        dataset = build_dataset(name)
        return dataset