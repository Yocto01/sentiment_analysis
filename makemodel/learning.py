import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
import files
from sentence_transformers import SentenceTransformer

# MPS (MacのGPU) が使えるかチェック
if torch.backends.mps.is_available():
    device = torch.device("mps")
    #print("MacのGPU (MPS) が有効です！🚀")
else:
    device = torch.device("cpu")
    #print("GPUが使えないため、CPUを使用します。")

model = SentenceTransformer("intfloat/multilingual-e5-base")
#print("Sentence-BERTモデルの読み込みが完了しました。")

database_vectors = np.load(files.DATABASE_VECTORS)
x_train = torch.tensor(database_vectors, dtype=torch.float32).view(-1, 768).to(device)

data = pd.read_csv(files.DATASET_FILE, header=None, names=['category', 'example'])
y_train_list = [int(y) - 1 for y in data['category'].tolist()]
y_train = torch.tensor(y_train_list, dtype=torch.long).to(device)
#print("学習データの読み込みが完了しました。")


testcase = []
correct_ans = []
with open(files.TESTDATA_FILE) as f:
    reader = csv.reader(f)
    for row in reader:
        correct_ans.append(int(row[0]) - 1)
        testcase.append(row[1])
embed_test = model.encode(testcase)
x_test = torch.tensor(embed_test, dtype=torch.float32).view(-1, 768).to(device)
#print("テスト用データの読み込みが完了しました。")
    
def evaluation(pre, cor):
    if len(pre) != len(cor):
        return -1
    
    # 実際のクラス数を動的に取得（+1は0始まりのため）
    num_classes = max(max(pre), max(cor)) + 1
    res = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    
    for i in range(len(pre)):
        # 0始まりに統一したため -1 は不要
        res[cor[i]][pre[i]] += 1
    
    return res


n_input = x_train.shape[1]          # 768入力
n_output = len(torch.unique(y_train))  # 4出力
n_hidden = 128      # 隠れ層のノード数

# モデルの定義
# 768入力4出力1隠れ層のニューラルネットワークモデル
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        self.l1 = nn.Linear(n_input, n_hidden)
        
        self.l2 = nn.Linear(n_hidden, n_output)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x_drop = self.dropout(x2)
        x3 = self.l2(x_drop)
        return x3       

# モデル変数の生成
# 乱数の固定化
SEED = 123 * 3
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

net = Net(n_input, n_output, n_hidden)
net = net.to(device)

# 学習率
lr = 0.01

# 損失関数：交差エントロピー関数
criterion = nn.CrossEntropyLoss()

# 最適化関数：勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)

# 繰り返し回数
num_epochs = 3000

# メインループ
for epoch in range(num_epochs):
    
    # 訓練フェーズ
    net.train()   # Dropout有効
    
    # 勾配の初期化
    optimizer.zero_grad()
    
    # 予測計算
    x_train = x_train.to(device)
    outputs = net(x_train)
    
    # 損失計算
    y_train = y_train.to(device)
    loss = criterion(outputs, y_train)
    
    # 勾配計算
    loss.backward()
    
    # パラメータ修正
    optimizer.step()

    # 予測ラベル算出
    predicted = torch.max(outputs, 1)[1]
    
    # 損失と精度の計算
    train_loss = loss.item()
    train_acc = (predicted == y_train).sum() / len(y_train)
    
    if ((epoch+1) % 100 == 0):        
        # 予測フェーズ
        net.eval()  # Dropout無効
        
        # 予測計算
        x_test = x_test.float().to(device)
        outputs_test = net(x_test)

        top_indices = torch.topk(outputs_test, 1, dim=1)[1]

        # NumPy配列に変換
        top_numpy = top_indices.cpu().numpy().flatten()
        
        result = evaluation(top_numpy, correct_ans)
        
        correct_count = sum(result[i][i] for i in range(len(result)))
        test_acc = correct_count / len(correct_ans)
        
        print(f'{epoch+1},{test_acc},{result}')
        


#print("学習が完了しました。予測フェーズに移行します。")

# 予測フェーズ

# 予測計算
x_test = x_test.to(device)
outputs_test = net(x_test)

# 上位3つのインデックスを取得
top_indices = torch.topk(outputs_test, 1, dim=1)[1]

# NumPy配列に変換
top_numpy = top_indices.cpu().numpy().flatten()


# 結果出力
for c in range(len(testcase)):
    print(f"{top_numpy[c]+1},{testcase[c]}")


torch.save(net.state_dict(), 'predict_model.pth')