text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

from transformers import pipeline

classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
print(classifier(text))