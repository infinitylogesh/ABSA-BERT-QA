import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils_glue import InputExample,convert_examples_to_features,_truncate_seq_pair,InputFeatures
from pytorch_transformers import BertForSequenceClassification,BertTokenizer,BertConfig
from quantize import QuantizedLayer,quantize
import pdb,numpy as np
import cPickle
import time,sys

labels = ["none", "negative","neutral","positive","conflict"]
categories = ["food","price","anecdotes","ambience","service"]

def load_artifacts(model_path,is_quantized=False):
    """ Loads pretrained model , tokenizer , config."""
    model_class = BertForSequenceClassification
    print("quantized_ouput/" if is_quantized else model_path)
    if not is_quantized:
        model = model_class.from_pretrained(model_path)
    else:
        model = torch.load("4bit_quantized_model.bin")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained("quantized_ouput/" if is_quantized else model_path)
    model.to("cpu")
    model.eval()
    return model,tokenizer,config

def convert_to_features(examples,tokenizer,cls_token_at_end=False,max_seq_length = 128, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=-1))
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    eval_dataloader = DataLoader(dataset,batch_size=8)
    for batch in eval_dataloader:
        batch = tuple(t.to("cpu") for t in batch)
        inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],  # XLM don't use segment_id
              }
    return inputs

def infer(model,inputs):
    outputs = model(**inputs)
    logits = outputs[:2][0]
    preds = logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    return preds

print(sys.argv)
is_quantized = True if int(sys.argv[1]) == 1 else False
model,tokenizer,config = load_artifacts("output/",is_quantized=is_quantized)
#model = torch.load("raw_pytorch_model.bin")
#model.to("cpu")
#model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters {} :".format(count_parameters(model)))

while True:
    review = str(raw_input("Enter review >> "))
    questions = ["what do you think of the {} of it ?".format(category) for category in categories]
    examples = [InputExample(guid=i, text_a=review, text_b=question, label=None) for i,question in enumerate(questions)]
    inputs = convert_to_features(examples,tokenizer)
    # cPickle.dump(inputs,open("sample_inputs.pkl",'w'))
    start_time = time.time()
    outputs = infer(model,inputs)
    print("time taken",time.time() - start_time)
    result = {category:labels[output] for category,output in zip(categories,outputs)}

    # pdb.set_trace()
    print(result)
