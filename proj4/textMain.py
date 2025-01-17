from transformers import pipeline

# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
classifier = pipeline(model="facebook/bart-large-mnli")

sentense = "I have a problem with my iphone that needs to be resolved asap!!"

preds = classifier(sentense, candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"])

print(preds)