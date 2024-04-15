# from transformers import pipeline

# transcriber = pipeline(task="automatic-speech-recognition")

# audio = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

# result = transcriber(audio)

# print(result)

from transformers import pipeline

vision_classifier = pipeline(model="google/vit-base-patch16-224")

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

preds = vision_classifier(images=image_url)

preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

print(preds)