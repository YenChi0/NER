import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import os


# 定義 LSTM 模型
class Net(nn.Module):
    def __init__(self, vocab_size, number_of_tags):
        super(Net, self).__init__()
        EMBEDDING_DIM, LSTM_HIDDEN_DIM = 50, 50  # 定義詞向量與 LSTM 隱藏層維度
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)  # 詞向量層
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_HIDDEN_DIM, batch_first=True)  # LSTM 層
        self.fc = nn.Linear(LSTM_HIDDEN_DIM, number_of_tags)  # 全連接層

    def forward(self, s):
        s = self.embedding(s)  # 轉換為詞向量
        s, _ = self.lstm(s)  # LSTM 運算
        s = s.view(-1, s.shape[2])  # 重新塑形，以適應全連接層
        s = self.fc(s)  # 全連接層輸出
        return F.log_softmax(s, dim=1)  # 使用 log softmax 以提高數值穩定性


# 定義損失函數

def loss_fn(outputs, labels):
    labels = labels.view(-1)
    mask = (labels >= 0).float()
    labels = labels % outputs.shape[1]
    num_tokens = int(torch.sum(mask))
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


# 計算準確率
def accuracy(outputs, labels):
    labels = labels.ravel()
    mask = (labels >= 0)
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(np.sum(mask))


if __name__ == "__main__":
    torch.manual_seed(230)

    sentences_path = r"C:\\Users\\clair\\OneDrive\\桌面\\專題爬蟲\\chromedriver_win32\\sentences(一句一行版).txt"
    labels_path = r"C:\\Users\\clair\\OneDrive\\桌面\\專題爬蟲\\chromedriver_win32\\labels(一句一行版).txt"

    # 讀取資料
    with open(sentences_path, "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip().split() for line in f]

    # 建立詞彙表與標籤表
    words = Counter(word for sentence in sentences for word in sentence)
    tags = Counter(tag for label in labels for tag in label)

    PAD_WORD, PAD_TAG, UNK_WORD = '<pad>', 'O', 'UNK'
    MIN_COUNT = 1
    words = [tok for tok, count in words.items() if count >= MIN_COUNT]
    tags = [tok for tok, count in tags.items() if count >= MIN_COUNT]

    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    words.append(UNK_WORD)

    model = Net(len(words), len(tags))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    vocab, tag_map = {x: i for i, x in enumerate(words)}, {x: i for i, x in enumerate(tags)}
    unk_ind, pad_ind = vocab[UNK_WORD], vocab[PAD_WORD]

    sentences = [[vocab.get(token, unk_ind) for token in sentence] for sentence in sentences]
    labels = [[tag_map[label] for label in label_seq] for label_seq in labels]
    # 印出標籤對應索引
    print("Tag to Index Mapping:")
    for tag, index in tag_map.items():
        print(f"{tag}: {index}")

    batch_max_len = max(len(s) for s in sentences)


    def get_batch(sentence, tags):
        batch_data = pad_ind * np.ones((1, batch_max_len))
        batch_labels = -1 * np.ones((1, batch_max_len))
        batch_data[0][:len(sentence)] = sentence
        batch_labels[0][:len(sentence)] = tags
        return torch.tensor(batch_data, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)


    for _ in range(10):
        total_acc, total_loss = 0.0, 0.0
        for i in range(len(sentences)):
            train_batch, labels_batch = get_batch(sentences[i], labels[i])
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            acc_rate = accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_acc += acc_rate
            total_loss += loss.item()
        print(f"loss: {total_loss / len(sentences):.2f}... acc_rate = {total_acc / len(sentences):.2f}")

    print("finished training!")

    total_loss, total_acc = 0.0, 0.0
    for s_index in range(len(sentences)):
        train_batch, labels_batch = get_batch(sentences[s_index], labels[s_index])
        output_batch = model(train_batch)
        predict = [torch.argmax(each_ner).item() for each_ner in output_batch]
        # loss = loss_fn(output_batch, labels_batch)
        # acc_rate = accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
        print(f"guess count = {len(predict)}, ans count= {labels_batch.size(1)}")
        print(f"guess  = {predict}\nanslst = {labels_batch.tolist()[0]}")
    #     total_loss += loss.item()
    #     total_acc += acc_rate
    # print(f"loss = {total_loss / len(sentences):.2f} ... acc_rate = {total_acc / 10:.2f}")

    for i in range(len(sentences)):
        train_batch, labels_batch = get_batch(sentences[i], labels[i])
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        acc_rate = accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_acc += acc_rate
        total_loss += loss.item()
    print(f"loss: {total_loss / len(sentences):.2f}... acc_rate = {total_acc / len(sentences):.2f}")
