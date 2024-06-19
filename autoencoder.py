import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from logadempirical.models.utils import ModelOutput


class AutoEncoder(nn.Module):
    def __init__(
            self,
            hidden_size=100,
            num_layers=2,
            num_directions=2,
            embedding_dim=16,
            vocab_size=20,
            num_classes=10  # 新增参数，类别数
    ):
        super(AutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes  # 存储类别数
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=(self.num_directions == 2),
        )

        self.encoder = nn.Linear(
            self.hidden_size * self.num_directions, self.hidden_size // 2
        )

        self.decoder = nn.Linear(
            self.hidden_size // 2, self.hidden_size * self.num_directions
        )
        self.classifier = nn.Linear(self.hidden_size // 2, self.num_classes)  # 新增分类层
        self.criterion = nn.MSELoss(reduction="none")

        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))
        self.sequence_length = 100

    def attention_net(self, lstm_output, device):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device='cuda'):
        x = features['sequential'].to(device)
        x = self.embedding(x)
        outputs, hidden = self.rnn(x.float())

        representation = outputs[:, -1, :]  # 取最后一个时间步的输出作为表示
        x_internal = self.encoder(representation)
        x_recst = self.decoder(x_internal)
        reconstruction_loss = self.criterion(x_recst, representation).mean()

        logits = self.classifier(x_internal)
        probabilities = F.softmax(logits, dim=-1)
        classification_loss = F.cross_entropy(logits, features['label'].to(device))  # 假设有标签

        total_loss = reconstruction_loss + classification_loss  # 总损失为重建损失和分类损失之和

        return ModelOutput(logits=logits, probabilities=probabilities, loss=total_loss, embeddings=representation)


if __name__ == '__main__':
    model = AutoEncoder(
        hidden_size=128,
        num_directions=2, num_layers=2, embedding_dim=300).cuda()
    inp = torch.rand(64, 10, 300)
    out = model(inp)
    print(out['y_pred'].shape)
