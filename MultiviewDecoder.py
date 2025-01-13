import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiviewLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, self_attention=None):
        super(MultiviewLayer, self).__init__()
        if self_attention:
            self.self_attn = self_attention
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        tgt = tgt.transpose(0, 1)
        
        return tgt

class MultiviewDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MultiviewDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer() for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

if __name__ == "__main__":
    # Example usage
    d_model = 512
    nhead = 8
    num_layers = 6
    decoder_layer = MultiviewLayer(d_model, nhead)
    transformer_decoder = MultiviewDecoder(decoder_layer, num_layers)

    # Dummy inputs
    batch_size = 32
    tgt_seq_len = 10
    memory_seq_len = 20
    tgt = torch.rand(batch_size, tgt_seq_len, d_model)  # (batch_size, tgt_seq_len, d_model)
    memory = torch.rand(batch_size, memory_seq_len, d_model)  # (batch_size, memory_seq_len, d_model)

    output = transformer_decoder(tgt, memory)
    print(output.shape)  # Should be (batch_size, tgt_seq_len, d_model)