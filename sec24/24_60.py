from gensim.models import KeyedVectors
model_dir = 'G:\entity_vector.model.bin'
model = KeyedVectors.load_word2vec_format(model_dir, binary=True)

results = model.most_similar(positive=['[イチロー]','[サッカー]'],negative=['[野球]'])
for result in results:
    print(result)
