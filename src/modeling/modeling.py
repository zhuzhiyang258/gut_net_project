import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging

RANDOM_SEED = 42

class GUTNetConfig:
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 1)
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.intermediate_size = kwargs.get('intermediate_size', 3072)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.attention_probs_dropout_prob = kwargs.get('attention_probs_dropout_prob', 0.1)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.num_classes = kwargs.get('num_classes', 2)
        self.problem_type = kwargs.get('problem_type', None)
        self.log_level = kwargs.get('log_level', "INFO")
        self.classifier_dropout = kwargs.get('classifier_dropout', None)
        self.network_type = kwargs.get('network_type', 'without_attention')
        self.seq_len = kwargs.get('seq_len', 128)

class GUTNetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.network_type == 'with_attention':
            torch.manual_seed(RANDOM_SEED)
            self.variable_embeddings = nn.Embedding(config.seq_len, config.hidden_size)
        self.value_embeddings = nn.Linear(config.input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.config.network_type == 'with_attention':
            # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
            batch_size, seq_len = x.shape
            feature_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(x.device)
            embedded_features = self.variable_embeddings(feature_indices)
            value_embeddings = self.value_embeddings(x.unsqueeze(-1))
            embeddings = embedded_features * value_embeddings
        else:
            # Original behavior for 'without_attention'
            embeddings = self.value_embeddings(x)
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class GUTNetSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class GUTNetSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GUTNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GUTNetSelfAttention(config)
        self.output = GUTNetSelfOutput(config)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class GUTNetIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class GUTNetOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GUTNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GUTNetAttention(config)
        self.intermediate = GUTNetIntermediate(config)
        self.output = GUTNetOutput(config)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class GUTNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([GUTNetLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states

class GUTNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = GUTNetEmbeddings(config)
        self.encoder = GUTNetEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        print(f"Input shape in GUTNet: {x.shape}")
        if self.config.network_type == 'without_attention':
            embedding_output = self.embeddings(x)
            encoder_outputs = self.encoder(embedding_output.unsqueeze(1))
            sequence_output = encoder_outputs.squeeze(1)
        else:
            embedding_output = self.embeddings(x)
            encoder_outputs = self.encoder(embedding_output)
            sequence_output = encoder_outputs
        
        print(f"Sequence output shape: {sequence_output.shape}")
        pooled_output = self.pooler_activation(self.pooler(sequence_output))
        
        return sequence_output, pooled_output

@dataclass
class SequenceClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class GUTNetForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_classes
        self.config = config

        self.gutnet = GUTNet(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        self.post_init()

    def post_init(self):
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, x, labels=None):
        print(f"Input shape in GUTNetForSequenceClassification: {x.shape}")
        outputs = self.gutnet(x)
        
        sequence_output, pooled_output = outputs
        print(f"Pooled output shape: {pooled_output.shape}")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        print(f"Logits shape: {logits.shape}")

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output,
            attentions=None
        )

    def print_model_architecture(self):
        logging.info("GUTNet Model Architecture:")
        logging.info(f"Input Dimension: {self.config.input_dim}")
        logging.info(f"Hidden Size: {self.config.hidden_size}")
        logging.info(f"Number of Hidden Layers: {self.config.num_hidden_layers}")
        logging.info(f"Number of Attention Heads: {self.config.num_attention_heads}")
        logging.info(f"Intermediate Size: {self.config.intermediate_size}")
        logging.info(f"Number of Labels: {self.config.num_labels}")
        logging.info(f"Problem Type: {self.config.problem_type}")
        logging.info("\nModel Structure:")
        logging.info(self)

# Set up logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Create configuration
    config = GUTNetConfig(
        input_dim=11, 
        num_labels=5, 
        problem_type="single_label_classification", 
        log_level="DEBUG",
        network_type="with_attention",  # Change this to test different network types
        seq_len=128
    )

    # Initialize model
    model = GUTNetForSequenceClassification(config)

    # Print the model architecture
    model.print_model_architecture()

    # Prepare input
    batch_size = 32
    seq_length = config.seq_len
    
    if config.network_type == 'with_attention':
        x = torch.randn(batch_size, seq_length)
    else:
        x = torch.randn(batch_size, config.input_dim)
    
    labels = torch.randint(0, config.num_labels, (batch_size,))

    # Forward pass
    outputs = model(x, labels=labels)

    # Access results
    loss = outputs.loss
    logits = outputs.logits

    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")

