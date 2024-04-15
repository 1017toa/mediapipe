#1
from sentence_transformers import SentenceTransformer, util
#2
model = SentenceTransformer("all-MiniLM-L6-v2")

#3
# Two lists of sentences
sentences1 = "The cat sits outside"
sentences2 = "a cat sits inside"

#4
# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#5
# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
print("{} \t\t {} \t\t Score: {:.4f}".format(
        sentences1, sentences2, cosine_scores[0][0]
    ))

# for i in range(len(sentences1)):
#     print("{} \t\t {} \t\t Score: {:.4f}".format(
#         sentences1[i], sentences2[i], cosine_scores[i][i]
#     ))