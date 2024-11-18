import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNText(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, kernel_sizes, num_channels, pretrained_embed=None, static=False, multichannel=False):
        super(CNNText, self).__init__()

        if multichannel: # multichannel
            self.embed_static = nn.Embedding.from_pretrained(torch.tensor(pretrained_embed, dtype=torch.float), freeze=True)
            self.embed_nonstatic = nn.Embedding.from_pretrained(torch.tensor(pretrained_embed, dtype=torch.float), freeze=False)
            self.embed_static.weight.requires_grad = False  # static은 파라미터 학습하지 않음
            self.embed_nonstatic.weight.requires_grad = True  # nonstatic은 파라미터 학습
            in_channels = embed_dim * 2
        else:
            if pretrained_embed is not None: # static or non-static
                self.embed = nn.Embedding.from_pretrained(torch.tensor(pretrained_embed, dtype=torch.float), freeze=static)
                if not static: # non-static
                    self.embed.weight.requires_grad = True
            else:
                self.embed = nn.Embedding(vocab_size, embed_dim)  
                nn.init.uniform_(self.embed.weight, a=-0.25, b=0.25) # random initialization
            in_channels = embed_dim
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channels, out_channels=num_channels, kernel_size=K, stride=1) for K in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, class_num)
        self.multichannel = multichannel

    def forward(self, x):
        if self.multichannel: # multichannel 모드, 두 개의 임베딩을 결합
            x_static = self.embed_static(x).permute(0, 2, 1).float()
            x_nonstatic = self.embed_nonstatic(x).permute(0, 2, 1).float()
            x = torch.cat((x_static, x_nonstatic), dim=1)  # [batch_size, 2*embed_dim, seq_len]
        else:
            # single channel 모드
            x = self.embed(x).permute(0, 2, 1).float()
        
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(item, kernel_size=item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

