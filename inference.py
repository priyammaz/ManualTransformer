
import cupy as cp
import pickle
from model import GPT2Config, get_gpt2

### Load tokenizer ###
with open("work_dir/char_to_idx.pkl", "rb") as f:
    char_to_idx = pickle.load(f)
with open("work_dir/idx_to_char.pkl", "rb") as f:
    idx_to_char = pickle.load(f)

vocab_size = len(char_to_idx)

### Load model ###
config = GPT2Config()
model = get_gpt2(config)
model.load("work_dir/character_transformer.npz")
model.eval()

### Generation function ###
def generate_text(model, start_text="h", length=256, temperature=1.0):
    seed = cp.array([[char_to_idx[ch] for ch in start_text]], dtype=cp.int32)
    generated = seed.flatten().tolist()

    for _ in range(length):
        curr_len = min(seed.shape[1], config.max_seq_len)
        causal_mask = cp.triu(cp.ones((1, 1, curr_len, curr_len)) * -cp.inf, k=1)
        logits = model.forward(seed[:, -config.max_seq_len:], causal_mask)
        last_logits = logits[-1]  # last position logits

        # Apply temperature & softmax
        logits_scaled = last_logits / temperature
        probs = cp.exp(logits_scaled - cp.max(logits_scaled))
        probs /= cp.sum(probs)

        next_token = cp.random.choice(vocab_size, size=1, p=probs)[0].get().item()
        generated.append(next_token)

        print(idx_to_char[next_token], end="", flush=True)

        seed = cp.array(generated, dtype=cp.int32).reshape(1, -1)

    return "".join([idx_to_char[i] for i in generated])


### Example usage ###
if __name__ == "__main__":
    text = generate_text(model, start_text="h", length=5000, temperature=0.8)