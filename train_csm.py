from datasets import load_dataset, Audio
from transformers import (
    CsmForConditionalGeneration,
    TrainingArguments,
    CsmProcessor,
    Trainer
)

# model_id = 'sesame/csm-1b'
model_id = '/nfs/pretrained_models/csm-1b'
# dataset_path = 'eustlb/dailytalk-conversations-grouped'
dataset_path = '/nfs/datasets/dailytalk-conversations-grouped'

processor = CsmProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id)
model.train()
model.codec_model.eval()

ds = load_dataset(dataset_path, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

def data_collator(samples):
    conversations = [] 

    for sample in samples:
        concatenated_audio_array = sample["audio"]["array"]
        audio = [concatenated_audio_array[s: e] for s, e in sample["audio_cut_idxs"]]
            
        conversation = []
        for speaker_id, text, audio in zip(sample["speaker_ids"], sample["texts"], audio):
            conversation.append({
                "role": f"{speaker_id}",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio": audio}
                ]
            })
            
        conversations.append(conversation)

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        output_labels=True,
    )
    return inputs

training_args = TrainingArguments(
    "test-trainer",
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model, 
    training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()
