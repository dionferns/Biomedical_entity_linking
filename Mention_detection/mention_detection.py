import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple


# Model used for mention detection
# Cross Entropy Loss()
    # Each sample belongs to one and only one class
    # Softmax activation over all classes where Only one class can be correct
    # For Prediction = argmax over all logits
    # Example: logits = [3.2, 1.1, -2.5]  # for [B, I, O]
    # CrossEntropyLoss(target=0)  # only one correct label (e.g., B)
# Why this model:
    # since for mention detection each token can only be one class(b, I or O)


# This is the label index to ignore when calculating cross entropy loss
# IGNORE_INDEX = -1

IGNORE_INDEX = cel_default_ignore_index = CrossEntropyLoss().ignore_index
# nn.CrossEntropyLoss() has a default parameter: ignore_index = -100, hence IGNORE_INDEX = CrossEntropyLoss().ignore_index
# is just equal to IGNORE_INDEX = -100.

# CrossEntropyLoss().ignore_index returns -100 (default).
#This constant tells the loss function to ignore specific positions, (tokens)
# in the sequence, e.g. padding tokens or special tokens like [CLS], [SEP].
# BUT WHY? Because the MentionDetection class takes in embeddings of the document not the original document as input, 
# hence it will already have tokens like [CLS] for classification etc, which we need to remove. 
# why do we need to remove them? to prevent noise, if we keep them, model will be penalised for mistakes of meaningless tokens

class MentionDetection(nn.Module):
    def __init__(self, num_labels: int, dropout: float, hidden_size: int):
        """
        Based on transformer's BertForTokenClassification class. Token
        :param num_labels: number of labels used for token
        :param dropout: dropout for linear layer
        :param hidden_size: embedding size
        """
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        # Randomly sets some elements of an input token to zero, Helps model to generalise better(reduce overfitting)
        self.linear = nn.Linear(hidden_size, num_labels)
        # hidden_size(dim of a token in a sentence), cahnges the dim of the token to the dim = num of labels, using linear transformation.
        self.init_weights()
        # If you called init_weights() before defining self.linear, the model wouldn't know that layer exists — so it wouldn't initialize its weights!

    def init_weights(self):
        """Initialize weights for all member variables with type nn.Module"""
        self.apply(self._init_weights)
        # the _init_weghts is applies to all the layers. 

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, contextualised_embeddings: torch.Tensor = None, ner_labels: torch.Tensor = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # contextualised_embeddings: type: Tensor, shape: [B, T, H] = The output of the Transformer encoder. For each token in a batch, you get vector of size H. 
        # B = Batch size, T = sentence length(number of tokens ), H = hidden size(H) for each token(e.g., 768 for BERT).

        # ner_lables: has shape [B, T]
        # ner_labels are the gold standard for the embeddings used to check the performance of the models in BIO tagging.
        
        """
        Forward pass of Mention Detection layer.
        Treated as token classification problem where each token is assigned a BIO label.
        B = Begin entity (number usually used: 1)
        I = In entity (number usually used: 2)
        O = Not an entity (number usually used: 0)
        IGNORE_INDEX (-1) is used in `bio_label` positions to mark tokens to exclude from the loss function
        :param contextualised_embeddings: encoder's contextualised token embeddings
        :param ner_labels: BIO labels for tokens (length matches sequence_output, including special tokens)
        :return: loss tensor (if ner_labels is provided), token activations tensor
        """

        contextualised_embeddings = self.dropout(contextualised_embeddings)
        logits = self.linear(contextualised_embeddings)
        # this makes the shape go from [B,T, H] -> [B, T, num_labels]   
        if ner_labels is not None:
            active_loss = ner_labels.view(-1) != IGNORE_INDEX
            # This returns a boolean tensor of shape [B*T] — it's True for positions that should be included in the loss, and False for [CLS], [SEP], [PAD], etc., that use IGNORE_INDEX.
            # ner_labels.view(-1) flattens the tensor from [B,T] to [B*T]

            active_logits = logits.view(-1, self.num_labels) # converts [B,T, C(number of labels)] -> [B*T, C](2D)
            # This is done since cross_entropy expects the logits shape to be [N, C](2D tensor)(N=number of tokens for all combined sentences, C = number of labels for each token)

            # Have to do this since CrossEntropyLoss expects inputs of shape [N, C] (N = number of samples, C = number of classes), and targets of shape [N]
            # So we convert: logits: from [B, T, num_labels] → [B*T, num_labels].

            # ignore loss from tokens where `bio_label` is -1
            active_labels = torch.where(
                active_loss, ner_labels.view(-1), torch.tensor(IGNORE_INDEX).type_as(ner_labels)    #.type_as ensures that the data type (dtype) of the IGNORE_INDEX tensor matches the data type of ner_labels
            )
            # torch.where(condition, x, y): where the ouput is either x[i] if the condition[i] is true or y[i] if the condition[i] is not true.

            # Cross entropy expects labels to have shape of [N], number of tokens from all the sentences, hence ner_labels.view(-1) used.
            loss = F.cross_entropy(active_logits, active_labels, ignore_index=IGNORE_INDEX)
            return loss, logits
            # loss: shape is a single number, logits: shape is [B, T, num_labels]
        return None, logits



