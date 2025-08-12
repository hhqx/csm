import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio

model_id = "/nfs/pretrained_models/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
conversation = []

# 1. context
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

# 2. text prompt
conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "outputs/example_with_context.wav")
