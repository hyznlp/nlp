from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer.encode(sequence_a)
assert len(encoded_sequence_a) == 8

encoded_sequence_b = tokenizer.encode(sequence_b)
assert len(encoded_sequence_b) == 19

padded_sequence_a = tokenizer.encode(sequence_a, max_length=19, pad_to_max_length=True)
padded_sequence_b = tokenizer.encode(sequence_b, max_length=19, pad_to_max_length=True)


assert padded_sequence_a == [101, 1188, 1110, 170, 1603, 4954,  119, 102,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,   0]
assert encoded_sequence_b == [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]

sequence_a_dict = tokenizer.encode_plus(sequence_a, max_length=19, pad_to_max_length=True)
sequence_b_dict = tokenizer.encode_plus(sequence_b, max_length=19, pad_to_max_length=True)


assert sequence_a_dict['input_ids'] == [101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert sequence_a_dict['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]