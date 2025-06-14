from PIL import Image
import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel


SAVE_DIR = "C:/Users/sarda/Documents/FYP Proj/OCRFULL/model"

processor = NougatProcessor.from_pretrained(SAVE_DIR)
model = VisionEncoderDecoderModel.from_pretrained(SAVE_DIR, torch_dtype=torch.bfloat16)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
context_length = model.decoder.config.max_position_embeddings
torch_dtype = model.dtype

def predict(img_path):
    image = Image.open(img_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(torch_dtype).to(device)

    outputs = model.generate(
        pixel_values,
        repetition_penalty=1.5,
        min_length=1,
        max_new_tokens=context_length,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )

    page_sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return page_sequence
