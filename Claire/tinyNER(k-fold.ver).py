import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import KFold


# 定義 LSTM 模型
class Net(nn.Module):
    def __init__(self, vocab_size, number_of_tags):
        super(Net, self).__init__()
        EMBEDDING_DIM, LSTM_HIDDEN_DIM = 50, 50
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(LSTM_HIDDEN_DIM, number_of_tags)

    def forward(self, s):
        s = self.embedding(s)
        s, _ = self.lstm(s)
        s = s.view(-1, s.shape[2])
        s = self.fc(s)
        return F.log_softmax(s, dim=1)


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

    sentences_path = "C:\\Users\\clair\\OneDrive\\桌面\\專題爬蟲\\chromedriver_win32\\sentences(一句一行版).txt"
    labels_path = "C:\\Users\\clair\\OneDrive\\桌面\\專題爬蟲\\chromedriver_win32\\labels(一句一行版).txt"

    with open(sentences_path, "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip().split() for line in f]

    words = Counter(word for sentence in sentences for word in sentence)
    tags = Counter(tag for label in labels for tag in label)

    PAD_WORD, PAD_TAG, UNK_WORD = '<pad>', 'O', 'UNK'
    MIN_COUNT = 1
    words = [tok for tok, count in words.items() if count >= MIN_COUNT]
    tags = [tok for tok, count in tags.items() if count >= MIN_COUNT]

    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    words.append(UNK_WORD)

    vocab, tag_map = {x: i for i, x in enumerate(words)}, {x: i for i, x in enumerate(tags)}
    unk_ind, pad_ind = vocab[UNK_WORD], vocab[PAD_WORD]

    sentences = [[vocab.get(token, unk_ind) for token in sentence] for sentence in sentences]
    labels = [[tag_map[label] for label in label_seq] for label_seq in labels]

    batch_max_len = max(len(s) for s in sentences)


    def get_batch(sentence, tags):
        batch_data = pad_ind * np.ones((1, batch_max_len))
        batch_labels = -1 * np.ones((1, batch_max_len))
        batch_data[0][:len(sentence)] = sentence
        batch_labels[0][:len(sentence)] = tags
        return torch.tensor(batch_data, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)


    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    total_acc, total_loss = 0.0, 0.0

    for fold, (train_idx, test_idx) in enumerate(kf.split(sentences)):
        print(f"Fold {fold + 1}")
        model = Net(len(words), len(tags))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 訓練
        for epoch in range(10):
            fold_acc, fold_loss = 0.0, 0.0
            for i in train_idx:
                train_batch, labels_batch = get_batch(sentences[i], labels[i])
                output_batch = model(train_batch)
                loss = loss_fn(output_batch, labels_batch)
                acc_rate = accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                fold_acc += acc_rate
                fold_loss += loss.item()
            print(
                f"Epoch {epoch + 1}: loss = {fold_loss / len(train_idx):.2f}, acc_rate = {fold_acc / len(train_idx):.2f}")

        # 測試
        fold_test_loss, fold_test_acc = 0.0, 0.0
        for i in test_idx:
            test_batch, labels_batch = get_batch(sentences[i], labels[i])
            output_batch = model(test_batch)
            predict = [torch.argmax(each_ner).item() for each_ner in output_batch]
            loss = loss_fn(output_batch, labels_batch)
            acc_rate = accuracy(output_batch.data.cpu().numpy(), labels_batch.data.cpu().numpy())
            fold_test_loss += loss.item()
            fold_test_acc += acc_rate
            print(f"Test Sample {i}:\nPredict: {predict}\nActual: {labels_batch.tolist()[0]}")
        print(
            f"Fold {fold + 1} Test Loss: {fold_test_loss / len(test_idx):.2f}, Test Acc: {fold_test_acc / len(test_idx):.2f}\n")

        total_acc += fold_test_acc / len(test_idx)
        total_loss += fold_test_loss / len(test_idx)

    print(f"Overall Loss = {total_loss / 10:.2f}, Overall Accuracy = {total_acc / 10:.2f}")
