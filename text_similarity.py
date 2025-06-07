#%%
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = SentenceTransformer('pretrained_frameworks/all-MiniLM-L6-v2')  # Small, fast, accurate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hg_model_hub_name = "pretrained_frameworks/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
roberta_model.to(device)

#%%

def text_sim(model, des1, des2):
    embedding1 = model.encode(des1, convert_to_tensor=True)
    embedding2 = model.encode(des2, convert_to_tensor=True)
    score = util.cos_sim(embedding1, embedding2)
    return score

def check_entailment(entailment_model, hypothesis, premise):
    '''
    hypothesis: editing prompt
    premise: image description
    '''
    tokenized_input_seq_pair = tokenizer.encode_plus(
        premise,
        hypothesis,
        max_length=512,
        return_token_type_ids=True,
        truncation=True
    )

    # Convert inputs to tensors and move to device
    input_ids = torch.tensor(tokenized_input_seq_pair['input_ids'], device=device).unsqueeze(0)
    attention_mask = torch.tensor(tokenized_input_seq_pair['attention_mask'], device=device).unsqueeze(0)
    token_type_ids = torch.tensor(tokenized_input_seq_pair['token_type_ids'], device=device).unsqueeze(0)

    outputs = entailment_model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    entailment = predicted_probability[0]
    neutral = predicted_probability[1]
    contradiction = predicted_probability[2]
    return entailment, neutral, contradiction

def region_loss(text_sim_model, entailment_model,
                org_description, edited_description, editing_prompt_adjusted):
    desc_sim = text_sim(text_sim_model, org_description, edited_description)
    desc_dist = 1 - desc_sim
    print(desc_dist)
    
    ent,neu,con = check_entailment(entailment_model, editing_prompt_adjusted, edited_description)
    entailment = max(ent,neu)
    contradiction = con
    print(ent, neu, con)
    
    if entailment < contradiction:
        loss = 0.5 * desc_dist + 0.5 * contradiction
    else:
        loss = desc_dist
    return float(loss)

#%%
s1 = '''
The image depicts a man seated on the floor in an indoor setting, likely a public space such as a train station or airport. He is dressed in a beige coat, black pants, and black shoes, with a dark turtleneck sweater visible underneath. His hair is neatly styled, and he has a beard. The man is focused on his laptop, which is open on his lap, suggesting he is working or studying. His legs are crossed, and he is holding his chin thoughtfully, indicating deep concentration. 
To his right, there is a large, gray backpack with brown straps, resting on the floor. The backpack appears to be made of a durable fabric, possibly canvas, and has a classic design with buckles and loops. The man's feet are flat on the ground, and his shoes have a rugged sole, suitable for walking or standing for extended periods. The floor is a polished, reflective surface, possibly made of stone or concrete, and it reflects the ambient light, adding to the overall brightness of the scene.
The background features large windows with a grid pattern, allowing natural light to flood the space. The light creates a warm, golden hue, casting soft shadows and highlighting the textures of the man's clothing and the backpack. The windows are framed by metal or wood, adding to the industrial feel of the environment. The overall mood of the image is one of quiet productivity and contemplation, with the man absorbed in his task amidst the serene, well-lit surroundings.
'''
s2 = '''
The image depicts an indoor setting with a man seated on the floor, leaning against a wall. He is wearing a beige coat, black pants, and dark shoes. His posture is relaxed, with one leg crossed over the other, and he is focused on his laptop screen. The man has short, neatly styled hair and a beard. His left hand is near his mouth, suggesting deep thought or concentration. 
To his right, there is a large, gray backpack with brown straps, resting on the floor. The backpack appears to be made of a durable fabric, possibly canvas, and has a classic design with buckles and loops. The floor is a polished, reflective surface, likely made of stone or tile, and it reflects the ambient light, creating a bright and airy atmosphere. The background features large windows with a grid pattern, allowing natural light to flood the space. The overall mood is calm and contemplative, with the man seemingly absorbed in his work or activity.
'''
edit_prompt = "Have the man cross his legs."
print(region_loss(model, roberta_model, s1, s2, edit_prompt))

# %%
