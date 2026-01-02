import os
print("Current working directory:", os.getcwd())
print("Vectorizer path:", os.path.abspath("data/processed/vectorizer.pkl"))
print("SVC model path:", os.path.abspath("data/processed/svc_model.pkl"))


import pickle
with open("../data/processed/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("../data/processed/svc_model.pkl", "rb") as f:
    svc_model = pickle.load(f)