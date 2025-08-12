
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, CsmProcessor, MimiModel
from datasets import load_dataset, Audio

model_id = "/nfs/pretrained_models/csm-1b"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

processor: CsmProcessor = AutoProcessor.from_pretrained(model_id)

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

conversation = []
# prepare a conversation with text and corresponding audio
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(torch_device)

model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
output = model(**inputs)

# model.codec_model: MimiModel = model.codec_model
# output.loss.backward()

print('loss:', output.loss)


