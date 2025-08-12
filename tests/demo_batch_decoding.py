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
# here a batch with two prompts
conversation = [
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
                {"type": "audio", "path": ds[0]["audio"]["array"]},
            ],
        },
        {
            "role": f"{ds[1]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[1]["text"]},
            ],
        },
    ],
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(device)

audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, [f"outputs/speech_batch_idx_{i}.wav" for i in range(len(audio))])
