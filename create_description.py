from datasets import load_dataset
import torch
import os
import sys
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from PIL import Image

sys.path.append(os.path.abspath('./pretrained_frameworks/LLMs/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

model_path = "pretrained_frameworks/LLMs/DeepSeek-VL2/pretrained_models/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer_vl = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

model_name = "pretrained_frameworks/LLMs/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
llm_model.generation_config = GenerationConfig.from_pretrained(model_name)
llm_model.generation_config.pad_token_id = llm_model.generation_config.eos_token_id

data_path = "downloaded_datatset/HumanEdit"
dataset = load_dataset(data_path, split="train", streaming=True)

output_path = "description_results.json"

if os.path.exists(output_path):
    os.remove(output_path)


def append_record_to_json_array(record, output_path):
    # Try to read existing array
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(record)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_batch(img_id, org_img, edit_prompt):
    llm_prompt1 = (
        " <image>\n Please describe this image in a detailed and comprehensive paragraph. "
        "Mention all visible objects, their relative positions, and any noticeable actions or interactions. "
        "Include descriptions of the background, setting (indoor or outdoor), lighting, colors, and materials. "
        "Comment on the mood or atmosphere, possible time of day, and any cultural or contextual significance. "
        "Use vivid language and full sentences, as if explaining the image to someone who cannot see it."
    )
    conversation = [
        {
            "role": "<|User|>",
            "content": llm_prompt1,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[org_img],
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer_vl.eos_token_id,
        bos_token_id=tokenizer_vl.bos_token_id,
        eos_token_id=tokenizer_vl.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    org_description = tokenizer_vl.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    llm_prompt2 = (
        f"Original description:\n{org_description}\n\n"
        f"Edit instruction:\n{edit_prompt}\n\n"
        "Modify the original description only to the extent necessary to reflect the edit instruction. "
        "Preserve the original sentence structure, phrasing, and wording unless changes are required to fulfill the instruction. "
        "Do not add, infer, or assume any new information beyond what the instruction clearly requires. "
        "Return the revised description."
    )

    messages = [
        {"role": "user", "content": llm_prompt2}
    ]

    # Tokenize using chat template (handles roles and formatting)
    input_tensor = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )

    # Generate the model response
    outputs = llm_model.generate(
        input_tensor.to(llm_model.device), 
        max_new_tokens=512
    )

    # Slice out the generated response only
    target_description = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True).strip()

    record = {
        "img_id": img_id,
        "edit_prompt": edit_prompt,
        "org_desc": org_description,
        "tar_desc": target_description
    }
    append_record_to_json_array(record, output_path)  # Save immediately

for sample in tqdm(dataset):
    image_id = sample["IMAGE_ID"]
    img = sample["INPUT_IMG"]
    editing_instruction = sample["EDITING_INSTRUCTION"]

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, bytes):
        import io
        img = Image.open(io.BytesIO(img)).convert("RGB")

    process_batch(image_id, img, editing_instruction)
        
# # %%
# index = 10
# llm_prompt1 = " <image>\n Please describe this image in a detailed and comprehensive paragraph. Mention all visible objects, their relative positions, and any noticeable actions or interactions. Include descriptions of the background, setting (indoor or outdoor), lighting, colors, and materials. Comment on the mood or atmosphere, possible time of day, and any cultural or contextual significance. Use vivid language and full sentences, as if explaining the image to someone who cannot see it."
# conversation = [
#     {
#         "role": "<|User|>",
#         "content": llm_prompt1,
#     },
#     {"role": "<|Assistant|>", "content": ""},
# ]

# #%%
# prepare_inputs = vl_chat_processor(
#     conversations=conversation,
#     images=[org_img[index].convert("RGB")],
#     force_batchify=True,
#     system_prompt=""
# ).to(vl_gpt.device)

# inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
# outputs = vl_gpt.language.generate(
#     inputs_embeds=inputs_embeds,
#     attention_mask=prepare_inputs.attention_mask,
#     pad_token_id=tokenizer_vl.eos_token_id,
#     bos_token_id=tokenizer_vl.bos_token_id,
#     eos_token_id=tokenizer_vl.eos_token_id,
#     max_new_tokens=512,
#     do_sample=False,
#     use_cache=True
# )

# answer = tokenizer_vl.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# print(answer)
# print("---------------------------------------------------")

# llm_prompt2 = (
#     f"Given the original description:\n{answer}\n\n"
#     f"Revise it only where necessary to reflect the specified editing instructions:\n{prompt[index]}\n\n"
#     "Preserve the original phrasing, sentence structure, and wording as much as possible. "
#     "Do not rephrase or add details beyond what is needed for the edit."
# )

# # Format as a chat message
# messages = [
#     {"role": "user", "content": llm_prompt2}
# ]

# # Tokenize using chat template (handles roles and formatting)
# input_tensor = tokenizer.apply_chat_template(
#     messages, 
#     add_generation_prompt=True, 
#     return_tensors="pt"
# )

# # Generate the model response
# outputs = llm_model.generate(
#     input_tensor.to(llm_model.device), 
#     max_new_tokens=512
# )

# # Slice out the generated response only
# result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True).strip()
# print(result)
