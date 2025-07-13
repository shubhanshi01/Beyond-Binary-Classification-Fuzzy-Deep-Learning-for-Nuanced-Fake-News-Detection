import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, membership_num):
        super(FuzzyLayer, self).__init__()
        self.input_dim = input_dim
        self.membership_num = membership_num
        self.membership_miu = nn.Parameter(torch.Tensor(membership_num, input_dim))
        self.membership_sigma = nn.Parameter(torch.Tensor(membership_num, input_dim))
        nn.init.xavier_uniform_(self.membership_miu)
        nn.init.ones_(self.membership_sigma)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        input_exp = input_seq.unsqueeze(1).expand(batch_size, self.membership_num, self.input_dim)
        miu_exp = self.membership_miu.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)
        sigma_exp = self.membership_sigma.unsqueeze(0).expand(batch_size, self.membership_num, self.input_dim)
        return torch.mean(torch.exp((-0.5) * ((input_exp - miu_exp) / sigma_exp) ** 2), dim=-1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, n_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, 32, kernel_size=1)
        self.rnn = nn.LSTM(32, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, metadata):
        embedded = self.dropout(F.relu(self.embedding(metadata))).unsqueeze(2)
        conved = F.relu(self.conv(embedded)).squeeze(2)
        outputs, (hidden, _) = self.rnn(conved.unsqueeze(0))
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

class LiarModel(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=128, n_filters=128,
                 filter_sizes=[3,4,5], output_dim=6, dropout=0.5, padding_idx=0,
                 input_dim=60, input_dim_metadata=6, hidden_dim=64, n_layers=1,
                 bidirectional=True):
        super().__init__()
        
        # TextCNN for statements
        self.textcnn = TextCNN(vocab_size, embedding_dim, n_filters, 
                             filter_sizes, output_dim, dropout, padding_idx)
        
        # TextCNN for justification
        self.justification_cnn = TextCNN(vocab_size, embedding_dim, n_filters,
                                       filter_sizes, output_dim, dropout, padding_idx)
        
        # TextCNN for metadata text
        self.textcnn2 = TextCNN(vocab_size, input_dim, n_filters,
                              filter_sizes, output_dim, dropout, padding_idx)
        
        # CNNBiLSTM for numerical metadata
        self.cnn_bilstm = CNNBiLSTM(input_dim_metadata, embedding_dim,
                                  hidden_dim, output_dim, n_layers,
                                  bidirectional, dropout)
        
        # Fuzzy layer
        self.fuzzy = FuzzyLayer(output_dim, output_dim)
        
        # Final fusion layer
        self.fuse = nn.Linear(output_dim * 5, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, metadata_text, metadata_number, justification):
        text_out = self.textcnn(text)
        just_out = self.justification_cnn(justification)
        meta_text_out = self.textcnn2(metadata_text)
        meta_num_out = self.cnn_bilstm(metadata_number)
        fuzzy_out = self.fuzzy(meta_num_out)
        
        combined = torch.cat((
            text_out, 
            just_out,
            meta_text_out,
            meta_num_out,
            fuzzy_out
        ), dim=1)
        
        return self.fuse(self.dropout(combined))

def load_model(model_path):
    # Initialize model with correct architecture
    model = LiarModel(
        vocab_size=30522,
        embedding_dim=128,
        n_filters=128,
        filter_sizes=[3,4,5],
        output_dim=6,
        dropout=0.5,
        padding_idx=0,
        input_dim=60,  # 6 metadata features * 10 dim each
        input_dim_metadata=6,
        hidden_dim=64,
        n_layers=1,
        bidirectional=True
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')