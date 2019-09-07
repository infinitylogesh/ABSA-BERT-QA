import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchtext import data
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from collections import OrderedDict
from handlers import Progbar
import itertools
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss, Accuracy, Precision, Recall 


def one_hot_aspects_queries(aspect_text):
    questions = [ 
    "what do you think of the price of it ?",
    "what do you think of the anecdotes of it ?",
    "what do you think of the food of it ?",
    "what do you think of the ambience of it ?",
    "what do you think of the service of it ?"
    ]
    for i,question in enumerate(questions):
        if aspect_text == question:
            return i

def one_hot_labels(aspect):
    labels =  ["none", "negative","neutral","positive","conflict"]
    return [1 if aspect == label else 0 for label in labels]


SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



TEXT = data.Field(tokenize = 'spacy')
ASPECT = data.Field(sequential=False,use_vocab=False,preprocessing=one_hot_aspects_queries)
LABEL = data.Field(sequential=False,use_vocab=False,preprocessing=one_hot_labels) 

train_logits = np.load("output_train_test_swapped/train_logits.npy")
test_logits = np.load("output_train_test_swapped/test_logits.npy")

train_data = data.TabularDataset(path="data/semeval-qm/train.tsv",format='tsv',fields=[("id",None),("label",LABEL),(("aspect",ASPECT)),(("text",TEXT))])
train_logits_data = TensorDataset(torch.from_numpy(train_logits).float())
test_logits_data = TensorDataset(torch.from_numpy(test_logits).float())


test_data = data.TabularDataset(path="data/semeval-qm/test.tsv",format='tsv',fields=[("id",None),("label",LABEL),(("aspect",ASPECT)),(("text",TEXT))])
TEXT.build_vocab(train_data,max_size = 80000, vectors = "glove.6B.300d", unk_init = torch.Tensor.normal_)
# LABEL.build_vocab(train_data)
train_iter = data.BucketIterator(train_data,batch_size=32,shuffle=False,train=True)
train_logits_iter = DataLoader(train_logits_data,batch_size=32,shuffle=False)
test_iter = data.BucketIterator(train_data,batch_size=32,shuffle=False,train=False)
test_logits_iter = DataLoader(test_logits_data,batch_size=32,shuffle=False)

#train_iter_2 = itertools.chain.from_iterable([train_iter,train_logits_iter])
#test_iter_2 = itertools.chain.from_iterable([test_iter,test_logits_iter])



def distillation_loss(student_logits,y_true,teacher_logits,temperature=1.0,alpha=0.5):
    
    def teacher_loss(student_logits,teacher_logits,temperature):
        soft_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        distillation_loss = F.kl_div(soft_log_probs, soft_targets.detach(), size_average=False) / soft_targets.shape[0]
        return distillation_loss
    
    def student_loss(student_logits,y_true):
        ce_loss = torch.nn.CrossEntropyLoss()
        student_loss = ce_loss(student_logits,torch.argmax(y_true,dim=-1))
        return student_loss
    
    return alpha * student_loss(student_logits,y_true) + (1-alpha)*teacher_loss(student_logits,teacher_logits,temperature)
    





class LSTMStudent(nn.Module):

    def __init__(self,embedding_dim,hidden_dim,vocab_size,target_size,num_aspects,aspect_emb_dimentions,bidirectional=False,num_lstm_layers=1,lstm_dropout=0.5):
        super(LSTMStudent,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.word_embeddings.weight = nn.Parameter(TEXT.vocab.vectors)
        self.word_embeddings.weight.requires_grad = False
        self.aspect_embeddings = nn.Embedding(num_aspects,aspect_emb_dimentions)
        self.word_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=num_lstm_layers,bidirectional=bidirectional,dropout=lstm_dropout)
        self.dropout = nn.Dropout(0.7)
        self.hidden2tag = nn.Linear((hidden_dim*num_lstm_layers)+aspect_emb_dimentions,target_size)
    
    def forward(self,sentence,aspect_query):
        embeds = self.word_embeddings(sentence)
        aspect_embedding = self.aspect_embeddings(aspect_query)
        _ , (hidden, cell) = self.lstm(embeds.transpose(0,1))
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        concat = torch.cat((hidden,aspect_embedding),dim=-1)
        return self.hidden2tag(concat)



def custom_supervised_evaluator(model,metrics,device,loss):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x,teacher_logits = batch
            teacher_logits = teacher_logits[0]
            text = x.text.to(device)
            aspect = x.aspect.to(device)
            y_true = x.label.to(device)
            y_pred = model(text.transpose(0,1),aspect)
            student_loss = distillation_loss(y_pred,y_true,teacher_logits=teacher_logits)
            return (y_pred,torch.argmax(y_true,dim=-1))

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def train(lstm):
    device = "cpu"
    train_loader = zip(train_iter,train_logits_iter)
    test_loader = zip(test_iter,test_logits_iter)
    optimizer = Adam(lstm.parameters(),0.001)
    ce_loss = torch.nn.CrossEntropyLoss()
    running_avgs = OrderedDict()
    def step(engine,batch):
        x,teacher_logits = batch
        teacher_logits = teacher_logits[0]
        text = x.text.to(device)
        aspect = x.aspect.to(device)
        y_true = x.label.to(device)
        n_batch = len(aspect)
        optimizer.zero_grad()
        y_pred = lstm(text.transpose(0,1),aspect)
        distill_loss = distillation_loss(y_pred,y_true,teacher_logits=teacher_logits)
        distill_loss.backward()
        optimizer.step()

        return {
            'distill_loss': distill_loss.item()
        }

    metrics = {
    'avg_accuracy': Accuracy(),
    'avg_precision': Precision(average=True), 
    'avg_recall': Recall(average=True)
    }
    
    trainer = Engine(step)

    train_evaluator = custom_supervised_evaluator(lstm, metrics=metrics, device=device,loss=ce_loss)
    val_evaluator = custom_supervised_evaluator(lstm, metrics=metrics, device=device,loss=ce_loss)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_offline_train_metrics(engine):
        epoch = engine.state.epoch
        metrics = train_evaluator.run(train_loader).metrics
        print("Training Results - Epoch: {}  Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f}"
            .format(engine.state.epoch, metrics['avg_accuracy'], metrics['avg_precision'], metrics['avg_recall']))
        
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_val_metrics(engine):
        epoch = engine.state.epoch
        metrics = val_evaluator.run(test_loader).metrics
        print("Validation Results - Epoch: {}  Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f}"
            .format(engine.state.epoch, metrics['avg_accuracy'], metrics['avg_precision'], metrics['avg_recall']))    

    checkpoint_handler = ModelCheckpoint("./", 'checkpoint',
                                         save_interval=3,
                                         n_saved=10, require_empty=False, create_dir=True)
    progress_bar = Progbar(loader=train_loader, metrics=running_avgs)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'net': lstm})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=progress_bar)
    trainer.run(train_loader, max_epochs=50)
    

lstm = LSTMStudent(300,256,80000,5,5,64,bidirectional=True,num_lstm_layers=2)
train(lstm)