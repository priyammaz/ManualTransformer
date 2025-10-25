"""
Training a tiny GPT type model very close to the MinGPT implementation from Karpathy!
https://github.com/karpathy/minGPT
"""
import cupy as cp
from tqdm import tqdm
import requests
from tqdm import tqdm
import pickle


from model import GPT2Config, get_gpt2

import nn
import optim 

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Dataset length: {len(text)} characters")

### Create A Tokenizer ###
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

### Save tokenizers ###
with open("work_dir/char_to_idx.pkl", "wb") as f:
    pickle.dump(char_to_idx, f)
with open("work_dir/idx_to_char.pkl", "wb") as f:
    pickle.dump(idx_to_char, f)

### Tokenize Data ###
data = cp.array([char_to_idx[ch] for ch in text])

### Get Batches ###
def get_batch(data, batch_size, seq_len):
    idx = cp.random.randint(0, len(data) - seq_len - 1, batch_size)
    inputs = cp.stack([data[i:i+seq_len] for i in idx])
    targets = cp.stack([data[i+1:i+seq_len+1] for i in idx])
    return inputs, targets.flatten()

### Load model (with default configs) ###
config = GPT2Config()
model = get_gpt2(config)

### Create a Causal Mask ###
causal_mask = cp.triu(cp.ones((1, 1, config.max_seq_len, config.max_seq_len)) * -cp.inf, k=1)

### Load Loss/Optimizer ###
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

### Train Model ###
model.train()
train_iterations = 5000
batch_size = 16

for epoch in tqdm(range(train_iterations)):
    inputs, targets = get_batch(data, batch_size, config.max_seq_len)

    ### Get Logits from Model ###
    logits = model.forward(inputs, causal_mask)
    
    ### Get Loss ###
    loss = loss_fn.forward(y_true=targets, logits=logits)
   
    ### Get derivative of loss w.r.t logits ###
    loss_grad = loss_fn.backward()

    ### Backprop those grads through the rest of the model ###
    model.backward(loss_grad)
    
    ### Update Model ###
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 500 == 0:
        preds = cp.argmax(logits, axis=-1)  # predicted indices
        accuracy = cp.mean(preds == targets) * 100  # percentage correct
        
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

        model.eval()
        seed = cp.array([[char_to_idx['h']]])  # starting character
        generated = [seed.item()]

        for _ in range(config.max_seq_len):
            curr_len = seed.shape[1]
            # causal_mask = cp.triu(cp.ones((1, 1, curr_len, curr_len)) * -1e9, k=1)

            logits = model.forward(seed, causal_mask[:, :, :curr_len, :curr_len])  # apply causal mask
            last_logits = logits[-1]  # take logits for last position

            # Softmax and sampling
            probs = cp.exp(last_logits - cp.max(last_logits))
            probs /= cp.sum(probs)

            next_token = cp.random.choice(vocab_size, size=1, p=probs)[0].get().item()  # sample 
            generated.append(next_token)
            seed = cp.array(generated).reshape(1, -1)
        
        print("".join([idx_to_char[i] for i in generated]))
        model.train()

model.save("work_dir/character_transformer.npz")

