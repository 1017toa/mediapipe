from transformers import pipeline

vqa = pipeline(model="impira/layoutlm-document-qa")

image_url = "https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png"

pred = vqa(image=image_url, question="What is the invoice number?")

print(pred)