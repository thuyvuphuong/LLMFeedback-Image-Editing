import os
import json
import torch
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath('./pretrained_frameworks/LLMs/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


def load_prompts(file_path):
    """Load prompts from the JSON file."""
    relations = []
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for item in data:
                relation = item.get("relation")
                relations.append(relation)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not valid JSON.")
    
    return relations

def make_description(image_path, scene_graph):
    llm_prompt = "Describe detailed in a paragraph about the objects and their relation in the given image <image>\n with the given information: " + scene_graph
    conversation = [
        {
            "role": "<|User|>",
            "content": llm_prompt,
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # Run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    return answer


#__________________________________Main__________________________________________________

model_path = "./pretrained_frameworks/LLMs/DeepSeek-VL2/pretrained_models/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

data_rootpath = "./pretrained_frameworks/SceneGraph/egtr/data/visual_genome/images"
with open("./pretrained_frameworks/SceneGraph/egtr/data/visual_genome/val.json", "r") as f:
    data = json.load(f)

file_names = [img["file_name"] for img in data.get("images", [])]
file_paths = [os.path.join(data_rootpath, file_names[i]) for i in range(len(file_names))]

scene_graph_output = load_prompts("./text_docs/scenegraph_output.json")

output_file = "text_docs/raw_description.json"

results = []
for i in tqdm(range(len(file_paths))):
    file_name = file_names[i]
    img_description = make_description(file_paths[i], scene_graph_output[i])
    output_data = {
        "file_names": file_name,
        "prompts": img_description
    }
    results.append(output_data)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
print(f"Results saved to {output_file}")