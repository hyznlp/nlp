# Albert+BI-LSTM+CRF的实体识别 Pytorch
### outline
![lstm_crf的模型结构](https://raw.githubusercontent.com/jiangnanboy/albert_lstm_crf_ner/master/pics/lstm_crf_layers.png)

**lstm_crf**

![albert_lstm的模型结构](https://raw.githubusercontent.com/jiangnanboy/albert_lstm_crf_ner/master/pics/albert_lstm.png)

**albert_embedding_lstm**

1.这里将每个句子split成一个个字token，将每个token映射成一个数字，再加入masks,然后输入给albert产生句子矩阵表示，比如一个batch=10，句子最大长度为126，加上首尾标志[CLS]和[SEP]，max_length=128,albert_base_zh模型输出的数据shape为(batch,max_length,hidden_states)=(10,128,768)。

2.利用albert产生的表示作为lstm的embedding层。

3.没有对albert进行fine-tune。

### train
hugging face albert


setp 2: 部分参数设置 models/config.yml

    embedding_size: 768
	hidden_size: 128
	model_path: models/
	batch_size: 64
	max_length: 128
	dropout: 0.5
	tags:
  		- ORG
  		- PER
  		- LOC
  		- T

step 3: train

    run main.py
	训练数据来自人民日报的标注数据

### evaluate
 	eval
	        ORG	recall 1.00	precision 1.00	f1 1.00
	        PER	recall 0.97	precision 0.96	f1 0.96
	        LOC	recall 1.00	precision 1.00	f1 1.00
	        T	recall 0.84	precision 0.80	f1 0.82
	




