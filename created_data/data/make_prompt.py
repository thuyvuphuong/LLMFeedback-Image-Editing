import torch
import os
import sys
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath('./pretrained_frameworks/LLMs/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

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

batch_size = 8  # Define the batch size
system_prompt = ""

def process_batch(batch_images):
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n Given the image, provide specific edit suggestions (in only complex sentence) for only 1 to 5 random visible objects, including replacements, enhancements, removal, additions, or applying stylistic changes, and should be described in a clear and natural sentence.",
            "images": batch_images,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=system_prompt
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
    
    return [tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True) for output in outputs]

output_file = "./text_docs/visualgenome_imageediting_data.json"
results = []

for i in tqdm(range(len(file_names))):
    batch_files = file_names[i:i+batch_size]
    batch_paths = file_paths[i:i+batch_size]
    try:
        batch_results = process_batch(batch_paths)
        for file_name, result in zip(batch_files, batch_results):
            output_data = {
                "file_names": file_name,
                "prompts": result
            }
            results.append(output_data)

            # Save results continuously
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            # Optional: print the current progress
            # print(f"Image: {file_name}, Suggestion: {result}")
            
    except Exception as e:
        print(f"Error processing batch {batch_files}: {e}")

print(f"Results saved to {output_file}")