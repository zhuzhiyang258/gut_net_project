import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging

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
        self.model_type = kwargs.get('model_type', 'without_attention')
        self.max_variables = kwargs.get('max_variables', 512)  # Renamed from max_position_embeddings

class GUTNetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Add this line to store the config
        if config.model_type == 'with_attention':
            self.variable_embeddings = nn.Embedding(config.max_variables, config.hidden_size)
        self.value_embeddings = nn.Linear(config.input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.config.model_type == 'with_attention':
            seq_length = x.size(1)
            variable_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            variable_embeddings = self.variable_embeddings(variable_ids)
            embeddings = variable_embeddings.unsqueeze(0) * x.unsqueeze(-1)
        else:
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
        self.debug = True  # Add this line

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if len(x.shape) == 3:
            return x.permute(1, 0, 2)  # 对于 3D 输入 即直接以变量的值为输入时的做法 此时attention不起作用，
        elif len(x.shape) == 4:
            return x.permute(0, 2, 1, 3)  # 对于 4D 输入 将变量嵌入后作为输入
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

    def forward(self, hidden_states):
        if self.debug:
            print(f"Hidden states shape in self-attention: {hidden_states.shape}")
            self.debug = False  # Turn off debug after first forward pass
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        if len(context_layer.shape) == 3:
            context_layer = context_layer.permute(1, 0, 2).contiguous()
        elif len(context_layer.shape) == 4:
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
        self.debug = True  # Add this line

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
        if self.debug:
            print(f"Input shape in GUTNet: {x.shape}")
        if self.config.model_type == 'with_attention':
            if len(x.shape) != 2:
                raise ValueError("Input should be 2D (batch_size, seq_len) when model_type is 'with_attention'")
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
        embedding_output = self.embeddings(x)
        if self.debug:
            print(f"Embedding output shape: {embedding_output.shape}")
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        if self.debug:
            print(f"Pooled output shape: {pooled_output.shape}")
            self.debug = False  # Turn off debug after first forward pass
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
        self.debug = True  # Add this line

    def post_init(self):
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, x, labels=None):
        if self.debug:
            print(f"Input shape in GUTNetForSequenceClassification: {x.shape}")
        outputs = self.gutnet(x)
        
        pooled_output = outputs[1]
        if self.debug:
            print(f"Pooled output shape: {pooled_output.shape}")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.debug:
            print(f"Logits shape: {logits.shape}")
            self.debug = False  # Turn off debug after first forward pass

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # 确保标签是 Long 类型
                labels = labels.long()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=None,
        )

    def print_model_architecture(self):
        logging.info("GUTNet Model Architecture:")
        logging.info(f"Input Dimension: {self.config.input_dim}")
        logging.info(f"Hidden Size: {self.config.hidden_size}")
        logging.info(f"Number of Hidden Layers: {self.config.num_hidden_layers}")
        logging.info(f"Number of Attention Heads: {self.config.num_attention_heads}")
        logging.info(f"Intermediate Size: {self.config.intermediate_size}")
        logging.info(f"Number of Classes: {self.config.num_classes}")
        logging.info(f"Problem Type: {self.config.problem_type}")
        logging.info("\nModel Structure:")
        logging.info(self)

# Set up logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Create configuration
    config = GUTNetConfig(input_dim=1, num_classes=5, problem_type="single_label_classification", 
                          log_level="DEBUG", model_type="with_attention", max_variables=512)

    # Initialize model
    model = GUTNetForSequenceClassification(config)

    # Print the model architecture
    model.print_model_architecture()

    # Prepare input
    batch_size = 32
    seq_length = 11  # This now represents the number of variables
    x = torch.randn(batch_size, seq_length)  # Each variable has a single value
    labels = torch.randint(0, config.num_classes, (batch_size,))

    # Forward pass
    outputs = model(x, labels=labels)

    # Access results
    loss = outputs.loss
    logits = outputs.logits

    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")
