import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM
import logging

logging.basicConfig(level=logging.INFO)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
#bpe/wordpiece
tokenized_text = tokenizer.tokenize(text)
# tokenized_text1 = tokenizer.encode(text)

masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#encode会自动加上【cls】 和 【sep】
# indexed_tokens1 = tokenizer.encode(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

model1 = BertForMaskedLM.from_pretrained('bert-base-uncased')
model1.eval()

# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

with torch.no_grad():
    outputs = model1(tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs[0]
    print(encoded_layers)

predicted_index = torch.argmax(encoded_layers[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)























