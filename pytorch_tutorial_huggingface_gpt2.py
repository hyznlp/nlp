# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import logging
#
# logging.basicConfig(level=logging.INFO)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
# text = "Who was Jim Henson ? Jim Henson was a"
# # tokenized_text = tokenizer.tokenize(text)
# indexed_tokens = tokenizer.encode(text)
# tokens_tensor = torch.tensor([indexed_tokens])
#
# model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# model.eval()
# # tokens_tensor = tokens_tensor.to('cuda')
# # segments_tensors = segments_tensors.to('cuda')
# # model.to('cuda')
#
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     encoded_layers = outputs[0]
#     print(encoded_layers)
#
# predicted_index = torch.argmax(encoded_layers[0, -1, :]).item()
# predicted_token = tokenizer.decode(indexed_tokens + [predicted_index])
# print(predicted_token)


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("The Manhattan bridge")
context = torch.tensor([generated])
past = None

for i in range(100):
    # print(i)
    output, past = model(context, past=past)
    token = torch.argmax(output[..., -1, :])

    generated += [token.tolist()]
    context = token.unsqueeze(0)
    print(generated, context, tokenizer.decode(generated))

sequence = tokenizer.decode(generated)

print(sequence)
