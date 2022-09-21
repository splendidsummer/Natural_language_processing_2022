import requests
import json
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from transformers import BertForQuestionAnswering
from transformers import AdamW


def load_data(file_name):
    with open(file_name, 'rb') as f:
        data = json.load(f)
    return data


def parse_data(data):

    contexts = []
    questions = []
    answers = []

    for group in data['data']:
        for para in group['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions,  answers


def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two so we fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


# This func is used to mapping char start/end positions into start/end token position
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        # print('i: ', i)
        # print('answers[i][answer_start]', answers[i]['answer_start'])

        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class squadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # every item in encoding item including: input_ids; token_type_ids; attention_mask;
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_file = './data/'
train_file = data_file + 'train-v2.0.json'
dev_file = data_file + 'dev-v2.0.json'

train_data = load_data(train_file)
dev_data = load_data(dev_file)
train_contexts, train_questions, train_answers = parse_data(train_data)
dev_contexts, dev_questions, dev_answers = parse_data(dev_data)
add_end_idx(train_answers, train_contexts)
add_end_idx(dev_answers, dev_contexts)

# from_pretrained is class method of PreTrainedTokenizerBase
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# also with tokenizer.tokenize(text) method
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
dev_encodings = tokenizer(dev_contexts)

add_token_positions(train_encodings, train_answers)
add_token_positions(dev_encodings, dev_answers)

train_dataset = squadDataset(train_encodings)
dev_dataset = squadDataset(dev_encodings)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()


def train(epoches, data_loader):
    for epoch in range(epoches):

        loop = tqdm(data_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch + 1}')
            loop.set_postfix(loss=loss.item())


def test(data_loader):
    acc = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)

            # model inputs is different from training
            outputs = model(input_ids, attention_mask=attention_mask)

            # start_logits, end_logits in outputs -- from hugging face API 
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
            acc.append(((end_pred == end_true).sum() / len(end_pred)).item())

    acc = sum(acc) / len(acc)

    print("\n\nT/P\tanswer_start\tanswer_end\n")
    for i in range(len(start_true)):
        print(f"true\t{start_true[i]}\t{end_true[i]}\n"
              f"pred\t{start_pred[i]}\t{end_pred[i]}\n")


if __name__ == '__main__':
    train(epoches=2, data_loader=train_loader)
    test(data_loader=dev_loader)























