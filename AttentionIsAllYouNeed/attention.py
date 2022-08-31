import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.bert_tokenizer import BertTokenizer
from bertviz.neuron_view import show

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(conig.vocab_size, config.hidden_size)

def scaled_dot_product_attention(q, k, v, mask = None):
  dim_k = query.size(-1)
  scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dim_k)
  weights = F.softmax(scores, dim=1)
  return torch.bmm(weights, value)


class AttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super(AttentionHead, self).__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)

  def forward(self, hidden_state):
    attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
    return attn_outputs

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super(MultiHeadAttention, self).__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
      [
        AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
      ]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)

  def forward(self, hidden_state):
    x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
    x = self.output_linear(x)
    return x

# multihead_attn = MultiHeadAttention(config)
# attn_output = multihead_attn(inputs_embeds)
# attn_output.size()

class FeedForward(nn.Module):
  def __init__(self, config):
    super(FeedForward, self).__init__()
    self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, x):
    x = self.linear_1(x)
    x = self.gelu(x)
    x = self.linear_2(x)
    x = self.dropout(x)
    return x

# feed_forward = FeedForward(config)
# ff_outputs = feed_forward(attn_outputs)
# ff_outputs.size()

class TransformerEncoderLayer(nn.Module):
  def __init__(self, config):
    super(TransformerEncoderLayer, self).__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.attention = MultiHeadAttention(config)
    self.feed_forward = FeedForward(config)

  def forward(self, x):
    # Apply layer normalization and then copy input into query, key and value
    hidden_state = self.layer_norm_1(x)
    # Apply attention with a skip connection
    x = x + self.attention(hidden_state)
    # Apply feed-forward layer with a skip connection
    x = self.feed_forward(self.layer_norm_2(x))
    return x

# encoder_layer = TransformerEncoderLayer(config)
# input_embeds.shape, encoder_layer(inputs_embeds).size()

# POSITIONAL EMBEDDINGS

class Embeddings(nn.Module):
  def __init__(self, config):
    super(Embeddings, self).__init__()
    self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.layer_norm = nn.LayerNorm(config.hidden_size, eps = 1e-12)
    self.dropout = nn.Dropout()

  def forward(self, input_ids):
    # Create posiition IDs for input sequence
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    # Create token and position embeddings
    token_embeddings = self.token_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids)
    # Combine token and position embeddings
    embeddings = token_embeddings + position_embeddings
    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings

# embedding_layer = Embeddings(config)
# embedding_layer(inputs.input_ids).size()

class TransformerEncoder(nn.Module):
  def __init__(self, config):
    super(TransformerEncoder, self).__init__()
    self.embeddings = Embeddings(config)
    self.layers = nn.ModuleList(
      [
        TransformerEncoderLayer(config)
        for _ in range(config.num_hidden_layers)
      ]
    )

  def forward(self, inputs):
    embeddings = self.embeddings(inputs)
    for layer in self.layers:
      embeddings = layer(embeddings)
    return embeddings

# Adding a classification head

class TransformerForSequenceClassification(nn.Module):
  def __init__(self, config):
    super(TransformerForSequenceClassification, self).__init__()
    self.transformer = TransformerEncoder(config)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, x):
    x = self.transformer(x)
    x = self.dropout(x)
    x = self.classifier(x)
    return x

# config.num_labels = 3
# encoder_classifier = TransformerForSequenceClassification(config)
# encoder_classifier(inputs.input_ids).size()


    