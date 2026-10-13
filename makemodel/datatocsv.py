import files

with open(files.DATASET_FILE_JSON, "r") as f:    # ファイルオープン
    for line in f:      # 1行ずつ読み取る
        replaced = line.replace('\"', '')   # ダブルクオーテーションを消す
        replaced = replaced.replace(',', '')    # カンマを消す
        splited = replaced.split()      # スペースで区切ってリストにする
        if len(splited) == 2:       # 要素数が2なら
            if splited[0] == 'emotion-id:':     # 中カテゴリIDなら抽出
                category_id = splited[1]
        elif len(splited) == 1:
            if len(splited[0]) >= 5:        # 5文字以上で文章（事例）と認識
                print(f'{category_id},{splited[0]}')      # 標準出力（ファイルに書き込む際は「>」で流す