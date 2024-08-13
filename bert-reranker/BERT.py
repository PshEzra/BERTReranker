# import modules
import numpy as np
import torch
import xml.etree.ElementTree as et
import random
import pyterrier as pt
import pandas as pd
import json
import re
import math

from IPython.display import display
from xml.dom.minidom import parse, parseString
from pyterrier.measures import *
from pyterrier.model import add_ranks

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader

if not pt.started():
    pt.init(version = 'snapshot')

def add_label_column(run_df, qrels_df = None):
  """
  run_df => minimal [qid, docid, docno, rank, score]
  qrels_df => minimal [qid, docno, label]
  """
  assert qrels_df is not None, "qrels_df should not be empty"
  run_df = run_df.merge(qrels_df, on = ["qid", "docno"], how = "left")
  run_df["label"] = run_df["label"].fillna(0)
  run_df["label"] = run_df["label"].astype(int) # ensure labels are ints
  return run_df

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
  acc = accuracy_score(labels, predictions)
  return {
      'accuracy': acc,
      'f1': f1,
      'precision': precision,
      'recall': recall
  }

def train_bert(model, train_dataset, dev_dataset):
  
    if torch.cuda.is_available():
        device = torch.device(f"cuda")
        print("we will be using a GPU.")
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)
   
    #device = torch.device("cpu")
    #model.to(device)

    training_args = TrainingArguments(
      output_dir='/content/data/results', \
      num_train_epochs = 6, \
      per_device_train_batch_size = BATCH_SIZE, \
      per_device_eval_batch_size = BATCH_SIZE, \
      warmup_steps = 100, \
      weight_decay = 0.01, \
      evaluation_strategy = "steps", \
      eval_steps = 128)

    # Inisialisasi Trainer
    trainer = Trainer(
      model = model,
      args = training_args,
      train_dataset = train_dataset,
      eval_dataset = dev_dataset,
      compute_metrics = compute_metrics)

    trainer.train()
    return model

def bert_score(model, dataset, batch_size = BATCH_SIZE):
  if torch.cuda.is_available():
    device = torch.device(f"cuda")
    model.to(device)
  else:
    device = torch.device("cpu")
    model.to(device)

  preds = None
  nb_eval_steps = 0
  data_loader = DataLoader(dataset, \
                           batch_size = batch_size, \
                           num_workers = number_of_cpus, \
                           shuffle = False)
  for batch in data_loader:
    model.eval()
    with torch.no_grad():
      inputs = {'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)}
      if "token_type_ids" in batch:
        inputs['token_type_ids'] = batch['token_type_ids'].to(device)
      outputs = model(**inputs)
      probs = torch.softmax(outputs.logits, dim = 1)
      nb_eval_steps += 1
      scores = probs[:, 1] # probability of relevant
      if preds is None:
        preds = scores.detach().cpu().numpy().flatten()
      else:
        batch_predictions = scores.detach().cpu().numpy().flatten()
        preds = np.append(preds, batch_predictions, axis = 0)
  return preds

class DFDataset(Dataset):
  def __init__(self, df, tokenizer, idf_lib=None, idf_thresh=None, doc_total=None, *args, split, get_doc_fn, tokenizer_batch=100):
    '''Initialize a Dataset object.
    Arguments:
        samples: A list of samples. Each sample should be a
        tuple with (query_id, doc_id, <label>), where label is optional
        tokenizer: A tokenizer object from Hugging Face's Tokenizer lib.
        (need to implement encode_batch())
        split: a name for this dataset
        get_doc_fn: a function that maps a row into the text of the document
        tokenizer_batch: How many samples to be tokenized at once by the tokenizer object.
    '''
    self.tokenizer = tokenizer
    tokenizer.padding_side = "right"
    print("Loading and tokenizing %s dataset of %d rows ..." % (split, len(df)))
    assert len(df) > 0
    self.labels_present = "label" in df.columns
    query_batch = []
    doc_batch = []
    sample_ids_batch = []
    labels_batch = []
    self.store = {}
    self.processed_samples = 0
    self.idf_lib = idf_lib
    self.idf_thresh = idf_thresh
    self.doc_total = doc_total
    i = 0
    #print(df.columns)
    for indx, row in df.iterrows():
      query_batch.append(row["query"])
      doc_batch.append(get_doc_fn(row))
      sample_ids_batch.append(row["qid"] + "_" + row["docno"])
      if self.labels_present:
        labels_batch.append(row["label"])
      else:
        labels_batch.append(-1) # no labels; for testing, not for training
      if len(query_batch) == tokenizer_batch or i == len(df) - 1:
        self._tokenize_and_dump_batch(doc_batch, query_batch, labels_batch, sample_ids_batch)
        query_batch = []
        doc_batch = []
        sample_ids_batch = []
        labels_batch = []
      i += 1

  def _tokenize_and_dump_batch(self, doc_batch, query_batch, labels_batch,
                               sample_ids_batch):
    '''tokenizes and dumps the samples in the current batch
    It also store the positions from the current file into the samples_offset_dict.
    '''
    # Use the tokenizer object
    #batch_tokens = self.tokenizer.batch_encode_plus(list(zip(query_batch, doc_batch)), max_length=512, pad_to_max_length=True)
    batch_tokens = self.tokenizer.batch_encode_plus(list(zip(query_batch, doc_batch)), max_length=512, padding='max_length', truncation=True)
    for idx, (sample_id, tokens) in enumerate(zip(sample_ids_batch, batch_tokens['input_ids'])):
        assert len(tokens) == 512
        # BERT supports up to 512 tokens. batch_encode_plus will enforce this for us.
        # the original implementation had code to truncate long documents with [SEP]
        # or pad short documents with [0]
        segment_ids = batch_tokens['token_type_ids'][idx] if 'token_type_ids' in batch_tokens else None
        attention_mask = batch_tokens['attention_mask'][idx]
        
        new_mask = []
        
        if self.idf_lib != None:
            assert self.idf_lib != None and self.idf_thresh != None
            for i in range(len(tokens)):
                if tokens[i] not in self.idf_lib.keys():
                    new_mask.append(batch_tokens['attention_mask'][idx][i])
                    continue
                    
                if tokens[i] == 0:
                    break
                    
                df = self.idf_lib[tokens[i]]
                idf = math.log2(self.doc_total / len(df))
                
                if idf > self.idf_thresh:
                    new_mask.append(1)
                else:
                    new_mask.append(0)
        
        self._store(sample_id, tokens, new_mask, segment_ids, labels_batch[idx])
        self.processed_samples += 1

  def _store(self, sample_id, token_ids, attention_mask, segment_ids, label):
    self.store[self.processed_samples] = (sample_id, token_ids, attention_mask, segment_ids, label)

  def __getitem__(self, idx):
    _, input_ids, attention_mask, token_type_ids, label = self.store[idx]
    out_dict = {'input_ids': torch.tensor(input_ids, dtype = torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype = torch.long),
                'labels': torch.tensor([label], dtype = torch.long)}
    if token_type_ids != None:
      out_dict['token_type_ids'] = torch.tensor(token_type_ids, dtype = torch.long)
    if not self.labels_present:
      del out_dict['labels']
    return out_dict

  def __len__(self):
    return len(self.store)

class BERTPipeline(pt.Estimator):

  def __init__(self, model_name, *args,
    get_doc_fn = lambda row: row["text"],
    max_train_rank = None,
    max_valid_rank = None,
    cache_threshold = None,
    **kwargs):
    super().__init__(*args, **kwargs)
    
    if model_name == "distilbert-base-uncased":
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    else:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    
    self.max_train_rank = max_train_rank
    self.max_valid_rank = max_valid_rank
    self.get_doc_fn = get_doc_fn
    self.test_batch_size = BATCH_SIZE
    self.cache_threshold = cache_threshold
    self.cache_dir = None

  def make_dataset(self, res, *args, **kwargs):
    return DFDataset(res, *args, **kwargs)

  def fit(self, tr, qrels_tr, va, qrels_va):
    tr = add_label_column(tr, qrels_tr)
    va = add_label_column(va, qrels_va)

    #print(tr.head(20))

    if self.max_train_rank is not None:
      tr = tr[tr["rank"] < self.max_train_rank]
    if self.max_valid_rank is not None:
      va = va[va["rank"] < self.max_valid_rank]

    tr_dataset = self.make_dataset(tr, self.tokenizer, split = "train", get_doc_fn = self.get_doc_fn)
    assert len(tr_dataset) > 0
    va_dataset = self.make_dataset(va, self.tokenizer, split = "valid", get_doc_fn = self.get_doc_fn)
    assert len(va_dataset) > 0
    self.model = train_bert(self.model, tr_dataset, va_dataset)
    return self

  def transform(self, te):
    te_dataset = DFDataset(te, self.tokenizer, split = "test", get_doc_fn = self.get_doc_fn)
    scores = bert_score(self.model, te_dataset, batch_size = self.test_batch_size)
    assert len(scores) == len(te), "Expected %d scores, but got %d" % (len(te), len(scores))
    te["score"] = scores
    return add_ranks(te)

  def load(self, filename):
    self.model.load_state_dict(torch.load(filename), strict=False)

  def save(self, filename):
    state = self.model.state_dict(keep_vars=True)
    for key in list(state):
      if state[key].requires_grad:
        state[key] = state[key].data
      else:
        del state[key]
    torch.save(state, filename)