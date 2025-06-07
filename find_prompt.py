#%%
import torch
import os
import sys
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from PIL import Image
from datasets import load_dataset

torch.cuda.set_device(1)
print(f"CUDA visible device count: {torch.cuda.device_count()}")
print(f"Using: {torch.cuda.get_device_name(0)}")

sys.path.append(os.path.abspath('./pretrained_frameworks/LLMs/DeepSeek-VL2'))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

model_path = "pretrained_frameworks/LLMs/DeepSeek-VL2/pretrained_models/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer_vl = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

#%%
data_path = "downloaded_datatset/HumanEdit"
dataset = load_dataset(data_path, split="train")[:50]

orginal_images = dataset["INPUT_IMG"]
edited_images = dataset["OUTPUT_IMG"]
edit_instructions = dataset["EDITING_INSTRUCTION"]

index = 37

llm_prompt1 = f'''<image> \n
You are a meticulous visual analyst. Carefully examine the given image and describe it in a single, flowing paragraph (maximum 520 tokens). Focus on every visually observable detail—such as color, texture, material, size, shape, and spatial relationships. Do not use bullet points or lists.
Avoid assumptions or inferences about unseen factors (e.g., time of day, season, emotions, story). Describe only what is directly visible in the image.
Your paragraph must naturally include the following:
- A clear overview of the setting (e.g., indoor/outdoor, environment type, lighting conditions, background elements, overall mood)
- Detailed description of each major object: its appearance, color, material (wood, metal, fabric, etc.), texture (smooth, rough, shiny, soft, etc.), size (relative to others), and spatial position (e.g., foreground, center-left)
- If humans or animals are present, describe each individual separately in full detail. Include:
    - Hair, face, visible skin or fur, and accessories
    - Clothing (color, texture, material, style, condition)
    - Pose: the orientation and position of every visible body part (head, arms, legs, torso, hands, feet)
    - Describe their stance or motion only if clearly visible, grounded in what is seen
- For images with multiple people or animals, ensure each is described distinctly and thoroughly, woven into the flow of the paragraph
- Describe all supporting/background elements such as furniture, walls, ground, vegetation, or objects in the distance
- Clearly express spatial relationships between elements (e.g., in front of, behind, next to, overlapping, under)
- You must explicitly describe the visual features of each object or region targeted in the editing instruction: "{edit_instructions[index]}", separately. For example, if the instruction is "The girl bent and raised her two hands," then describe: The girl posture (e.g., leaning forward, bent knees) and The position and gesture of her hands (e.g., raised above shoulders, palms open)
Use vivid, sensory-rich language. Every detail must be grounded in what can actually be seen. Avoid summarizing—immerse the reader in a scene constructed entirely from the image visible content.
'''

conversation = [
    {
        "role": "<|User|>",
        "content": llm_prompt1,
        "images": ["test_imgs/test_img.jpg"]
    },
    {"role": "<|Assistant|>", "content": ""},
]

#%%
# orginal_images or edited_images
input_img = orginal_images[index].convert("RGB")
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=[input_img],
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

answer = tokenizer_vl.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(answer)

# %%
