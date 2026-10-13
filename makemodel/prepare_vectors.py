from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import files

model = SentenceTransformer("intfloat/multilingual-e5-base")

data = pd.read_csv(files.DATASET_FILE, header=None, names=['category', 'example'])

examples = data['example'].tolist()

embeddings = model.encode(examples)

np.save("database_vectors.npy", embeddings)
