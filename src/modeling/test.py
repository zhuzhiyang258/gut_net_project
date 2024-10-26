from transformers.models.bert import BertModel, BertConfig,BertForMultipleChoice
import torch


def run_bert():
    bert_config = BertConfig(
        vocab_size=32000,
        hidden_size=4096 // 2,
        intermediate_size=11008 // 2,
        num_hidden_layers=32 // 2,
        num_attention_heads=32 // 2,
        max_position_embeddings=2048 // 2,
    )

    bert_model = BertModel(config=bert_config)
    print(bert_model)

    input_ids = torch.randint(
        low=0,
        high=bert_config.vocab_size,
        size=(4, 30),
    )

    res = bert_model(input_ids)
    print(res)


if __name__ == "__main__":
    run_bert()
