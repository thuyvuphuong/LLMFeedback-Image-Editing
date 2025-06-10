#%%
import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

#%%
# Paths
data_path = "downloaded_datatset/HumanEdit"
json_path = "target_description.json"
model_name = "openai/clip-vit-base-patch32"

# %%
# Load img_id -> tar_desc mapping
with open(json_path, "r") as f:
    desc_data = json.load(f)
if isinstance(desc_data, list):
    imgid2desc = {item["img_id"]: item["tar_desc"] for item in desc_data}
else:
    imgid2desc = desc_data
    
# %%
dataset = load_dataset(data_path, split="train", streaming=True)

# CLIP processor and model
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# %%
class CLIPStreamDataset(IterableDataset):
    def __init__(self, hf_streaming_dataset, imgid2desc, processor):
        self.dataset = hf_streaming_dataset
        self.imgid2desc = imgid2desc
        self.processor = processor

    def __iter__(self):
        for example in self.dataset:
            img_id = example["IMAGE_ID"] if "IMAGE_ID" in example else example["img_id"]
            image = example["INPUT_IMG"] if "INPUT_IMG" in example else example["input_img"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            text = self.imgid2desc.get(img_id, "")
            proc = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            yield {
                "pixel_values": proc["pixel_values"].squeeze(0),
                "input_ids": proc["input_ids"].squeeze(0),
                "attention_mask": proc["attention_mask"].squeeze(0),
            }
            
# %%
batch_size = 32
num_epochs = 5
learning_rate = 5e-5

# %%
# DataLoader for streaming
stream_ds = CLIPStreamDataset(dataset, imgid2desc, processor)
loader = DataLoader(stream_ds, batch_size=batch_size)

# %%
# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = 10000  # Set this to an estimated total number of steps per epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps * num_epochs)

# %%
# Training loop
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, batch in enumerate(tqdm(loader, total=total_steps)):
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            return_loss=True,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        if step >= total_steps:
            break
    # Optionally save checkpoint
    torch.save(model.state_dict(), f"clip-vitb32-stream-epoch{epoch+1}.pt")

print("Training complete.")
# %%
