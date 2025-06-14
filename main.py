from PIL import Image
import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel

# Load the model and processor
processor = NougatProcessor.from_pretrained("MohamedRashad/arabic-large-nougat")
model = VisionEncoderDecoderModel.from_pretrained(
    "MohamedRashad/arabic-large-nougat",
    torch_dtype=torch.bfloat16,
    attn_implementation={"decoder": "flash_attention_2", "encoder": "eager"},
)

# Get the max context length of the model & dtype of the weights
context_length = model.decoder.config.max_position_embeddings
torch_dtype = model.dtype

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def predict(img_path):
    # prepare PDF image for the model
    image = Image.open(img_path)
    pixel_values = (
        processor(image, return_tensors="pt").pixel_values.to(torch_dtype).to(device)
    )

    # generate transcription
    outputs = model.generate(
        pixel_values.to(device),
        repetition_penalty=1.5,
        min_length=1,
        max_new_tokens=context_length,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )

    page_sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return page_sequence


#print(predict("C:\Users\sarda\Documents\FYP Proj\OCRFULL\Testimg\Surah-Al-Falaq-Arabic-and-Translation-graphic.jpg"))
