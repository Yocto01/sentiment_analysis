from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F # softmax用

app = Flask(__name__)

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

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

device = torch.device("cpu")

net = Net(n_input=768, n_output=4, n_hidden=128).to(device)

net.load_state_dict(torch.load('model/predict_model.pth', map_location=device))
net.eval()

sentiment_value = ['喜', '怒', '哀', '楽']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # フォームから送信されたテキストを取得 ('content' は textarea の name属性)
        user_text = request.form["content"]
        embed_user_text = embedding_model.encode([user_text])
        
        # numpy to tensor
        embed_user_text = embedding_model.encode([user_text])
        embed_tensor = torch.tensor(embed_user_text, dtype=torch.float32).to(device)
        
        # 推論
        with torch.no_grad(): # 推論時は勾配計算をオフにすると少し軽くなります
            outputs = net(embed_tensor)
            
        # softmaxを適用し確率を求める
        probs = F.softmax(outputs, dim=1)[0] * 100
        probs_list = probs.cpu().tolist()
        
        top_index = torch.topk(outputs, 1, dim=1)[1]

        # ----------------------------------------------------
        # ※ ここに後でAIモデルの推論処理（テキスト→ベクトル化→推論）を入れます
        # 今回はテストとして固定のダミー数値をHTMLに渡します
        # ----------------------------------------------------
        
        return render_template("index.html", 
                               user_input=user_text,
                               result_text=sentiment_value[top_index],
                               # 小数第1位で丸める
                               ki_percent=round(probs_list[0], 1),
                               do_percent=round(probs_list[1], 1),
                               ai_percent=round(probs_list[2], 1),
                               raku_percent=round(probs_list[3], 1))
    
    # 最初の画面（GETリクエスト）
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
