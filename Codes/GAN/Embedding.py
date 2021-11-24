import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, embedding_matrix, word_dropout_rate, isTrainEmbedding = True, padding = None):
        super(Embedding, self).__init__()
        self.vocab_size = embedding_matrix.shape[0]
        self.input_size = embedding_matrix.shape[1]

        if padding is not None:
            self.embedding = nn.Embedding(self.vocab_size, self.input_size, padding_idx = padding)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        self.EDropout = nn.Dropout(word_dropout_rate)

        if isTrainEmbedding == False:
            self.embedding.weight.requires_grad = False
            # for p in self.embedding.parameters():
            #     p.requires_grad = False

        self.embedding.weight.data.copy_(embedding_matrix)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.EDropout(embeddings)
        return embeddings