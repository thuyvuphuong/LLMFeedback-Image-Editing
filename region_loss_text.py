#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
torch.cuda.set_device(1)

# %%
max_length = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
hg_model_hub_name = "pretrained_frameworks/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
model.to(device)

# %%
data_path = "downloaded_datatset/HumanEdit"
dataset = load_dataset(data_path, split="train")
editing_instructions = dataset["EDITING_INSTRUCTION"]
output_captions = dataset["OUTPUT_CAPTION_BY_LLAMA"]

# %%
def check_entailment(hypothesis, premise):
    '''
    hypothesis: editing prompt
    premise: image description
    '''
    tokenized_input_seq_pair = tokenizer.encode_plus(
        premise,
        hypothesis,
        max_length=max_length,
        return_token_type_ids=True,
        truncation=True
    )

    # Convert inputs to tensors and move to device
    input_ids = torch.tensor(tokenized_input_seq_pair['input_ids'], device=device).unsqueeze(0)
    attention_mask = torch.tensor(tokenized_input_seq_pair['attention_mask'], device=device).unsqueeze(0)
    token_type_ids = torch.tensor(tokenized_input_seq_pair['token_type_ids'], device=device).unsqueeze(0)

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    entailment = predicted_probability[0]
    neutral = predicted_probability[1]
    contradiction = predicted_probability[2]
    return entailment, neutral, contradiction

#%%
hypothesis = "A person lowers his legs naturally"
premise = '''
The image captures a serene coastal scene where a woman stands waist-deep in the ocean, her body angled slightly to the left as she extends her right arm upward, creating a sense of movement and grace. She wears a vibrant, multicolored dress with intricate patterns that contrast beautifully against the soft blues and greens of the water and sky. Her dark hair is pulled back, revealing large hoop earrings that catch the light. The background features a rocky shoreline with waves crashing against the rocks, and palm trees swaying gently in the breeze. The sky is clear, with a few birds flying in the distance, adding a touch of life to the tranquil setting. The overall mood is one of peaceful solitude, enhanced by the warm tones and natural beauty of the environment
'''

check_entailment(hypothesis, premise)
# e, n, c = check_entailment(hypothesis, premise)
# print("Hypothesis:", hypothesis)
# print("Premise:", premise)
# print("Entailment:", e)
# print("Neutral:", n)
# print("Contradiction:", c)

#%%
entailments = []
neutrals = []
contradictions = []

for i in tqdm(range(len(editing_instructions))):
    entailment, neutral, contradiction = check_entailment(editing_instructions[i], output_captions[i])
    entailments.append(entailment)
    neutrals.append(neutral)
    contradictions.append(contradiction)
    
entailments = np.array(entailments)
neutrals = np.array(neutrals)
contradictions = np.array(contradictions)

#%%
max_entailment_or_neutral = np.maximum(entailments, neutrals)

# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

# Entailments histogram
axs[0].hist(entailments)
axs[0].set_title('Entailment Scores')
axs[0].set_xlabel('Score')
axs[0].set_ylabel('Frequency')
axs[0].grid(True, linestyle='--', alpha=0.6)

# Neutrals histogram
axs[1].hist(neutrals)
axs[1].set_title('Neutral Scores')
axs[1].set_xlabel('Score')
axs[1].grid(True, linestyle='--', alpha=0.6)

# Contradictions histogram
axs[2].hist(contradictions)
axs[2].set_title('Contradiction Scores')
axs[2].set_xlabel('Score')
axs[2].grid(True, linestyle='--', alpha=0.6)

axs[3].hist(max_entailment_or_neutral)
axs[3].set_title('Max of Entailment and Neutral')
axs[3].set_xlabel('Score')
axs[3].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
