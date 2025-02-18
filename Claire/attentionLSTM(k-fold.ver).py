import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold

# 設定超參數
BATCH_SIZE = 2
EPOCHS = 10
K_FOLDS = 10
LEARNING_RATE = 0.001

# 讀取 sentences.txt 和 labels.txt
sentences_path = r"C:\Users\clair\OneDrive\桌面\專題爬蟲\chromedriver_win32\sentences(一句一行版).txt"
labels_path = r"C:\Users\clair\OneDrive\桌面\專題爬蟲\chromedriver_win32\labels(一句一行版).txt"

with open(sentences_path, "r", encoding="utf-8") as f:
    sentences = [line.strip().split() for line in f.readlines()]

with open(labels_path, "r", encoding="utf-8") as f:
    tags = [line.strip().split() for line in f.readlines()]

# 建立詞彙表與標籤映射
token_set = set(word for sent in sentences for word in sent)
word_to_ix = {word: i + 1 for i, word in enumerate(token_set)}  # 轉換單詞到索引
word_to_ix["<PAD>"] = 0  # 設定填充標記（Padding token）

tag_set = set(tag for tag_seq in tags for tag in tag_seq)
tag_to_ix = {tag: i for i, tag in enumerate(tag_set)}  # 轉換標籤到索引
ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}  # 反向映射標籤索引
tag_to_ix["<PAD>"] = len(tag_to_ix)

PAD_IDX = word_to_ix["<PAD>"]  # 填充標記的索引
PAD_TAG_IDX = tag_to_ix["<PAD>"]


# 轉換句子和標籤為索引
def prepare_sequence(seq, to_ix):
    return [to_ix.get(w, to_ix["<PAD>"]) for w in seq]


train_data = [(prepare_sequence(sent, word_to_ix), prepare_sequence(tag_seq, tag_to_ix))
              for sent, tag_seq in zip(sentences, tags)]


# 定義批量處理函數
def collate_fn(batch):
    sentences, labels = zip(*batch)
    max_len = max(len(s) for s in sentences)
    sentences_padded = [s + [PAD_IDX] * (max_len - len(s)) for s in sentences]
    labels_padded = [l + [PAD_TAG_IDX] * (max_len - len(l)) for l in labels]
    return torch.tensor(sentences_padded, dtype=torch.long), torch.tensor(labels_padded, dtype=torch.long)


# 定義數據集類別
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 定義模型
class BiLSTMAttentionNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=128, attn_dim=64):
        super(BiLSTMAttentionNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_scores = self.fc(lstm_out)
        return tag_scores


# K-fold 交叉驗證
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
dataset = NERDataset(train_data)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{K_FOLDS}')

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMAttentionNER(len(word_to_ix), len(tag_to_ix))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 訓練模型
    for epoch in range(EPOCHS):
        model.train()
        for words, tags in train_loader:
            optimizer.zero_grad()
            outputs = model(words)
            outputs = outputs.view(-1, len(tag_to_ix))
            tags = tags.view(-1)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

    # 驗證模型並輸出預測結果
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for words, tags in val_loader:
            outputs = model(words)
            predictions = torch.argmax(outputs, dim=-1)

            for i in range(words.shape[0]):
                sentence_tokens = [list(word_to_ix.keys())[list(word_to_ix.values()).index(w.item())] for w in words[i]
                                   if w.item() != PAD_IDX]
                predicted_tags = [ix_to_tag[p.item()] for p in predictions[i] if p.item() != PAD_TAG_IDX]
                true_tags = [ix_to_tag[t.item()] for t in tags[i] if t.item() != PAD_TAG_IDX]
                print(f"Sentence: {' '.join(sentence_tokens)}")
                print(f"Predicted Tags: {' '.join(predicted_tags)}")
                print(f"True Tags: {' '.join(true_tags)}\n")

                for p_tag, t_tag in zip(predictions[i], tags[i]):
                    if t_tag != PAD_TAG_IDX:
                        total += 1
                        if p_tag == t_tag:
                            correct += 1

    accuracy = correct / total if total > 0 else 0
    fold_accuracies.append(accuracy)
    print(f'Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%')

# 輸出平均準確度
print(f'Average Accuracy over {K_FOLDS} folds: {np.mean(fold_accuracies) * 100:.2f}%')
