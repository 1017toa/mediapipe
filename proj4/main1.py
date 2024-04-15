from fastapi import FastAPI, Form
from transformers import pipeline
from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier2 = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
question_answerer = pipeline("question-answering", model="saudi82s/my_awesome_qa_model")
translator = pipeline("translation", model="stevhliu/my_awesome_opus_books_model")
generator = pipeline("text-generation", model="tzartrooper/my_awesome_eli5_clm-model")
mask_filler = pipeline("fill-mask", "stevhliu/my_awesome_eli5_mlm_model")

# summarizer(text)

app = FastAPI()

@app.post("/summarization/")
async def summarization(text: str = Form()):
    result = summarizer(text)
    # text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
    # "summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up workers and create good-paying, union jobs across the country."
    
    return {"result": result}

@app.post("/classification/")
async def classification(text: str = Form()):
    result = classifier(text)
    # text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    # [{'label': 'POSITIVE', 'score': 0.9994940757751465}]
    return {"result": result}

@app.post("/token_classification/", response_model=List[Entity])
async def token_classification(text: str = Form()):
    result = classifier2(text)
    # TypeError: can only concatenate str (not "list") to str
    # print("token_classification : " + result)
    # text = "The Golden State Warriors are an American professional basketball team based in San Francisco." 
    return result

@app.post("/causal_language/")
async def causal_language(text: str = Form()):
    result = generator(text)
    # prompt = "Somatic hypermutation allows the immune system to"
    return {"result": result}

@app.post("/masked_language/")
async def causal_language(text: str = Form()):
    result = mask_filler(text)
    # text = "The Milky Way is a <mask> galaxy."
    return {"result": result}

@app.post("/answering/")
async def answering(question : str = Form(), context  : str = Form()):
    result = question_answerer(question=question, context=context)
    # question = "How many programming languages does BLOOM support?"
    # context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

#     {'score': 0.2058267742395401,
#  'start': 10,
#  'end': 95,
#  'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
    return {"result": result}

@app.post("/translation/")
async def translation(text: str = Form()):
    result = translator(text)
    # text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
    # [{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.'}]    
    return {"result": result}