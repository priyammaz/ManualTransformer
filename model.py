from dataclasses import dataclass
import nn

@dataclass
class GPT2Config:
    vocab_size: int = 65
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    dim_mult: int = 4
    dropout_p: float = 0.1
    max_seq_len: int = 256

def get_gpt2(config):

    model = nn.NeuralNetwork()
    model.add(nn.Embedding(vocab_size=config.vocab_size, embed_dim=config.embed_dim))
    model.add(nn.PositionalEmbeddings(max_seq_len=config.max_seq_len, embed_dim=config.embed_dim))

    for _ in range(config.num_layers):
        model.add(nn.TransformerBlock(embed_dim=config.embed_dim, 
                                      num_heads=config.num_heads,
                                      dropout_p=config.dropout_p,
                                      dim_mult=config.dim_mult))

    model.add(nn.FlattenForLLM())
    model.add(nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size))

    return model

