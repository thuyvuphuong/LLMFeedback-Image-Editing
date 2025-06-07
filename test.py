import torch
import os
import sys
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath('./pretrained_frameworks/LLMs/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

model_path = "pretrained_frameworks/LLMs/DeepSeek-VL2/pretrained_models/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# scene_graph = "banana on plate, plate on table, banana in plate, plate above table, plate sitting on table, plate with banana, table under plate, banana near plate, plate in table, banana above plate, banana behind plate, plate of table, banana on table, banana on plate, plate has banana, plate laying on table, plate over table, table with plate, plate holding banana, plate under banana, table on plate, table has plate, plate at table, table holding plate, banana on plate, banana laying on table, banana above table, table near plate, plate wearing table, banana in plate, plate under table, plate and table, plate of banana, plate covered in table, plate with table, banana in table, plate has table, banana on plate, table in plate, plate on plate, plate on plate, plate on table, table of plate, banana on plate, banana in plate, table in front of plate, banana on plate, table carrying plate, banana sitting on table, banana on plate, table behind plate, plate with banana, table above plate, table holding banana, plate on table, plate on plate, table sitting on plate, banana in plate, banana in plate, plate sitting on table, banana in plate, plate above table, plate on plate, banana lying on table, plate of plate, table made of plate, banana on banana, banana in plate, table with banana, plate in plate, banana above plate, table under banana, plate with banana, plate sitting on table, plate of plate, plate under banana, plate above table, plate under plate, plate on plate, plate has banana, plate near plate, plate near plate, banana on table, banana near plate, plate of banana, banana laying on table, table under plate, plate of plate, banana on banana, plate with plate, plate in plate, banana covering table, plate near plate, plate of plate, handle on table, plate above plate, plate under plate, plate in table, plate with plate, plate in front of plate"
# llm_prompt = "Describe details in the given image <image>\n with the given information: " + scene_graph
llm_prompt = " <image>\n Please describe this image in a detailed and comprehensive paragraph. Mention all visible objects, their relative positions, and any noticeable actions or interactions. Include descriptions of the background, setting (indoor or outdoor), lighting, colors, and materials. Comment on the mood or atmosphere, possible time of day, and any cultural or contextual significance. Use vivid language and full sentences, as if explaining the image to someone who cannot see it."
conversation = [
    {
        "role": "<|User|>",
        "content": llm_prompt,
        "images": ["test_img.jpg"],
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

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
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
print(answer)