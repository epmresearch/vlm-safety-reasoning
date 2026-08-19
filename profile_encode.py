import time
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

texts = ["This is a test string to see how long it takes to encode on CPU."] * 256

print("Encoding 256 strings...")
start = time.time()
embs = model.encode(texts, convert_to_tensor=True, device="cpu")
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
