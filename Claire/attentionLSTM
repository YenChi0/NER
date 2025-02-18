import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
PAD_TAG_IDX = tag_to_ix["<PAD>"] # 填充標籤索引，若無"O"則新增
print(ix_to_tag)



# 轉換句子和標籤為索引
def prepare_sequence(seq, to_ix):
    return [to_ix.get(w, to_ix["<PAD>"]) for w in seq]  # 如果詞不在詞彙表內，填充為 <PAD>


train_data = [(prepare_sequence(sent, word_to_ix), prepare_sequence(tag_seq, tag_to_ix))
              for sent, tag_seq in zip(sentences, tags)]


# **批量處理：填充序列**
def collate_fn(batch):
    """ 讓 DataLoader 處理不同長度的句子，使用 padding """
    sentences, labels = zip(*batch)

    max_len = max(len(s) for s in sentences)  # 找出最長的句子長度

    # 進行填充
    sentences_padded = [s + [PAD_IDX] * (max_len - len(s)) for s in sentences]
    labels_padded = [l + [PAD_TAG_IDX] * (max_len - len(l)) for l in labels]

    return torch.tensor(sentences_padded, dtype=torch.long), torch.tensor(labels_padded, dtype=torch.long)


# 自訂數據集類別
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 數據加載器
BATCH_SIZE = 2  # 可以根據 GPU 調整 batch_size
train_loader = DataLoader(NERDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# **雙向 LSTM + 加性注意力模型**
class BiLSTMAttentionNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=128, attn_dim=64):
        super(BiLSTMAttentionNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)  # 嵌入層
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)  # 雙向 LSTM

        # 加性注意力機制
        self.attn_W = nn.Linear(hidden_dim * 2, attn_dim)  # W_a 權重矩陣
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)  # v_a 向量

        # 輸出層（分類器）
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def attention(self, lstm_output):
        """ 加性注意力機制 """
        attn_weights = torch.tanh(self.attn_W(lstm_output))  # (batch, seq_len, attn_dim)
        attn_weights = self.attn_v(attn_weights)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 計算注意力權重
        context = attn_weights * lstm_output  # 應用注意力
        context = context.sum(dim=1)  # 在時間維度上進行加總
        return context

    def forward(self, x):
        embeds = self.embedding(x)  # 嵌入層輸出
        lstm_out, _ = self.lstm(embeds)  # LSTM 輸出
        context = self.attention(lstm_out)  # 應用注意力機制
        tag_scores = self.fc(lstm_out)  # 預測每個時間步的標籤
        return tag_scores


# 建立模型實例
model = BiLSTMAttentionNER(vocab_size=len(word_to_ix), tagset_size=len(tag_to_ix))

# 定義損失函數與優化器
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX)  # 忽略填充標籤的損失計算
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練循環
EPOCHS = 10
for epoch in range(EPOCHS):
    for words, tags in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(words)  # 模型前向傳播

        outputs = outputs.view(-1, len(tag_to_ix))  # 重新調整形狀以計算損失
        tags = tags.view(-1)  # 展平標籤
        loss = criterion(outputs, tags)  # 計算損失

        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")


# 預測函數
def predict(sentences):
    predictions = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = prepare_sequence(sentence, word_to_ix)
            inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)  # 添加批次維度
            outputs = model(inputs)
            pred_tags = torch.argmax(outputs, dim=-1).squeeze().tolist()
            predictions.append([ix_to_tag[idx] for idx in pred_tags])
    return predictions


# **對 sentences.txt 中的所有句子進行預測**
pred_results = predict(sentences)

# 輸出結果
for sent, tags in zip(sentences, pred_results):
    print("Sentence:", " ".join(sent))
    print("Predicted Tags:", " ".join(tags))
    print()


# 計算準確度
def evaluate_accuracy(sentences, true_labels):
    correct = 0
    total = 0

    pred_results = predict(sentences)  # 取得預測結果

    for pred_tags, true_tags in zip(pred_results, true_labels):
        for p_tag, t_tag in zip(pred_tags, true_tags):
            if t_tag != "<PAD>":  # 忽略填充標籤
                total += 1
                if p_tag == t_tag:
                    correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


# 計算並顯示準確度
accuracy = evaluate_accuracy(sentences, tags)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
