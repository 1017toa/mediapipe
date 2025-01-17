question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

from transformers import pipeline

question_answerer = pipeline("question-answering", model="saudi82s/my_awesome_qa_model")
print(question_answerer(question=question, context=context))