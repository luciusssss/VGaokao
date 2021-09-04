from transformers.modeling_bert import *
import torch.nn.functional as F
from torch.nn import MarginRankingLoss

class BertHingeForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids_1=None,
        attention_mask_1=None,
        token_type_ids_1=None,
        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_1 = self.bert(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=token_type_ids_1,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output_1 = outputs_1[1]
        pooled_output_1 = self.dropout(pooled_output_1)
        logits_1 = self.classifier(pooled_output_1)
        prob_1 = F.softmax(logits_1)[:, 1]


        outputs_2 = self.bert(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output_2 = outputs_2[1]
        pooled_output_2 = self.dropout(pooled_output_2)
        logits_2 = self.classifier(pooled_output_2)
        prob_2 = F.softmax(logits_2)[:, 1]

        loss = None
        if labels is not None:
            loss_fct = MarginRankingLoss(margin=0.5)
            loss = loss_fct(prob_1.view(-1), prob_2.view(-1), labels.view(-1))
        # print('logit_1 shape:', logits_1.shape)
        # print('prob_1 shape:', prob_1.shape)
        if not return_dict:
            # output = (torch.cat([logits_1[:, 1], logits_2[:, 1]], -1),) + (outputs_1[2:], outputs_2[2:])
            output = (F.softmax(logits_1) - F.softmax(logits_2),) + outputs_1[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits= torch.cat([logits_1[:, 1], logits_2[:, 1]], -1),
            hidden_states=[outputs_1.hidden_states, outputs_2.hidden_states],
            attentions=[outputs_1.attentions, outputs_2.attentions],
        )