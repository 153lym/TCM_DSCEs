from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
from torch import nn
import torch.nn.functional as F

from crf import ChainCRF
import utils

class BertForSequentialClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels, tag_space=0, rnn_mode='LSTM',
                 use_crf=False,use_word_attention=False,
                 rnn_hidden_size=None, dropout=None):
        super(BertForSequentialClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        #self.bert.cuda()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if dropout is None:
            dropout = config.hidden_dropout_prob
        self.dropout_other = nn.Dropout(dropout)
        self.use_crf = use_crf
        self.use_word_attention = use_word_attention
        self.rnn_hidden_size = rnn_hidden_size

        if use_word_attention:
            natt = config.hidden_size
            att_num = 300
            self.W = nn.Linear(natt, att_num)
            self.V = nn.Parameter(torch.randn(att_num).float())
            self.softmax = nn.Softmax(dim=-1)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU

        if rnn_hidden_size is None:
            rnn_hidden_size = config.hidden_size // 2

        if use_word_attention:
            input_hiddensize = config.hidden_size * 2
            self.rnn = RNN(input_hiddensize, rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.rnn = RNN(config.hidden_size, rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        # BERT_有BiLSTM的
        out_dim = rnn_hidden_size * 2
        if tag_space:
            self.dense = nn.Linear(out_dim, tag_space)
            out_dim = tag_space
        else:
            self.dense = None
        if use_crf:
            self.crf = ChainCRF(out_dim, num_labels, bigram=True)
        else:
            self.dense_softmax = nn.Linear(out_dim, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, document_mask=None, labels=None):
        encoded_layers, output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        output = self.dropout(output)

        # sentence level transform to document level
        length = document_mask.sum(dim=1).long()
        max_len = length.max()
        output = output.view(-1, max_len, self.config.hidden_size)

        # document level RNN processing
        output, hx, rev_order, mask = utils.prepare_rnn_seq(output, length, hx=None,
                                                            masks=document_mask, batch_first=True)
        output, hn = self.rnn(output, hx=hx)
        output, hn = utils.recover_rnn_seq(output, rev_order, hx=hn, batch_first=True)

        # apply dropout for the output of rnn
        output = self.dropout_other(output)
        if self.dense is not None:
            # [batch, length, tag_space]
            output = self.dropout_other(F.elu(self.dense(output)))

        # final output layer
        if not self.use_crf:
            # not use crf
            output = self.dense_softmax(output) # [batch, length, num_labels]
            if labels is None:
                _, preds = torch.max(output, dim=2)
                return preds
            else:
                return (F.cross_entropy(output.view(-1, output.size(-1)), labels.view(-1), reduction='none') *
                        document_mask.view(-1)).sum() / document_mask.sum()
        else:
            #print(output.size())
            # CRF processing
            if labels is not None:
                return self.crf.loss(output, labels, mask=document_mask).mean()
            else:
                return self.crf.decode(output, mask=document_mask, leading_symbolic=0)

    """def forward(self, input_ids, token_type_ids=None, attention_mask=None, document_mask=None, labels=None):
        encoded_layers, output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        output = self.dropout(output)
        encode= self.dropout(encoded_layers)
        Hw = torch.tanh(self.W(encode))
        w_score = self.softmax(Hw.matmul(self.V))
        encode = encode.mul(w_score.unsqueeze(-1))
        att_out = torch.sum(encode, dim=1)
        # sentence level transform to document level
        length = document_mask.sum(dim=1).long()
        max_len = length.max()
        att_out = att_out.view(-1, max_len, self.config.hidden_size)
        output = output.view(-1, max_len, self.config.hidden_size)
        bilstm_in = torch.cat((output, att_out), 2)  

        output = bilstm_in

        # document level RNN processing
        output, hx, rev_order, mask = utils.prepare_rnn_seq(output, length, hx=None,
                                                            masks=document_mask, batch_first=True)
        output, hn = self.rnn(output, hx=hx)
        output, hn = utils.recover_rnn_seq(output, rev_order, hx=hn, batch_first=True)

        # apply dropout for the output of rnn
        output = self.dropout_other(output)

        if self.dense is not None:
            # [batch, length, tag_space]
            output = self.dropout_other(F.elu(self.dense(output)))

        # final output layer
        if not self.use_crf:
            # not use crf
            output = self.dense_softmax(output) # [batch, length, num_labels]
            if labels is None:
                _, preds = torch.max(output, dim=2)
                return preds
            else:
                return (F.cross_entropy(output.view(-1, output.size(-1)), labels.view(-1), reduction='none') *
                        document_mask.view(-1)).sum() / document_mask.sum()
        else:
            #print(output.size())
            # CRF processing
            if labels is not None:
                return self.crf.loss(output, labels, mask=document_mask).mean()
            else:
                return self.crf.decode(output, mask=document_mask, leading_symbolic=0)"""




