import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout) # embed = timeF
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.cls = nn.Linear(d_model, 3, bias=True) # 3 classes of the curve type
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, curve, dec_curve_inp=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, retrival = False, reduce_hid = False):

        self.pred_len = x_enc.shape[1]-1
        #self.pred_len = x_enc.shape[1]
        enc_out = self.enc_embedding(x_enc, x_mark_enc, curve)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # torch.Size([32, 25, 512])
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, reduce_hid = reduce_hid)

        if retrival:
            return enc_out, attns

        dec_out = self.dec_embedding(x_dec, x_mark_dec, dec_curve_inp)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_reg_out = self.projection(dec_out)
        dec_cls_out = self.cls(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], attns
        else:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], enc_out # [B, L, D]
    
    def forward_single(self, enc_out, x_dec, x_enc, x_mark_enc=None, x_mark_dec =None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, retrival = False, reduce_hid = False):

        self.pred_len = x_enc.shape[1]-1
        #self.pred_len = x_enc.shape[1]
        enc_out2 = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # torch.Size([32, 25, 512])
        enc_out2, attns = self.encoder(enc_out2, attn_mask=enc_self_mask, reduce_hid = reduce_hid)

        # if retrival:
        #     return enc_out, attns
        

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out2, x_mask=dec_self_mask, cross_mask=dec_enc_mask, reduce_hid=reduce_hid)
        dec_reg_out = self.projection(dec_out)
        dec_cls_out = self.cls(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], attns
        else:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], enc_out2 # [B, L, D]

    def forward_with_enc(self, enc_out, x_dec, x_enc, x_mark_enc=None, x_mark_dec =None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, retrival = False, reduce_hid = False):

        self.pred_len = x_enc.shape[1]-1
        
        enc_out2 = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # torch.Size([32, 25, 512])
        enc_out2, attns = self.encoder(enc_out2, attn_mask=enc_self_mask, reduce_hid = reduce_hid)
        #enc_out
        # if retrival:
        #     return enc_out, attns
        #with torch.no_grad():
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_reg_out = self.projection(dec_out)
        dec_cls_out = self.cls(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], attns
        else:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], enc_out2 # [B, L, D]
    
    def decode_exp(self, enc_out, x_dec, x_mark_dec=None, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, retrival = False, reduce_hid = False):

        self.pred_len = x_dec.shape[1]-1
        #self.pred_len = x_enc.shape[1]
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # torch.Size([32, 25, 512])
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, reduce_hid = reduce_hid)

#         if retrival:
#             return enc_out, attns

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_reg_out = self.projection(dec_out)
        dec_cls_out = self.cls(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:], attns
        else:
            return dec_reg_out[:,-self.pred_len:,:], dec_cls_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, retrival = False, reduce_hid = False):
        
        self.pred_len = x_enc.shape[1]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, reduce_hid = reduce_hid)

        if retrival:
            return enc_out, attns

        # bottle net -- font into one space 平移不变性 with VIT
        # position embedding

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, reduce_hid = reduce_hid)
        dec_out = self.projection(dec_out)
        
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
