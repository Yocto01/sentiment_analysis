from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools

model = SentenceTransformer(
    "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
) 

# 文章
sentences = [
        "勝手にブレーキを解除する",
        "ブレーキかけず立ち上がる",
        "勢いよく車椅子を漕ぎ出す",
        "周囲を確認しないで動かす",
        "危険なスピードで移動する",
        "車椅子操作が不安定である",
        "足を床に引きずり移動する",
        "片側のブレーキだけを外す",
        "介助を待たずに自ら動かす",
        "移乗する時に固定を忘れる",
        "車椅子の外へ身を乗り出す",
        "フットレストを外さず立つ",
        "操作中に注意が散漫になる",
        "坂道で操作しようと試みる",
        "車椅子から落ちそうになる",
        "ブレーキを繰り返しいじる",
        "見守りなしでの操作は危険",
        "足置きに乗って立ち上がる",
        "車椅子を押し歩こうとする",
        "浅く座るため姿勢が崩れる",
        "他の人の車椅子にぶつかる",
        "車椅子の上から無理に立つ",
        "車輪に手を巻き込まれそう",
        "肘掛けに寄りかかって傾く",
        "操作を誤り段差に突っ込む",
        "肘掛けの所に座ろうとする",
        "姿勢が崩れてずり落ちそう",
        "後ろを見ずに車椅子を漕ぐ",
        "スロープを勢いよく下りる",
        "車輪の隙間に指を入れそう"
]

embeddings = model.encode(sentences)
num_sentences = len(sentences) # 文章の数

sim_matrix = cosine_similarity(embeddings)  # コサイン類似度を計算

upper_loose = 0.75    # 上限(ゆるい)
upper_strist = 0.85  # 上限(きつい)
lower_loose = 0.15    # 下限(ゆるい)
lower_strict = 0.10  # 下限(きつい)
looser_candidates = []  # 候補([ i 番目, j 番目, 類似度]というリストが入る)
strict_candidates = [] # 候補(きつい条件)
loose_counts = [0] * num_sentences    # 候補に選ばれた回数( i 番目は i 番目の文章が候補に選ばれた回数が入る)
strict_counts = [0] * num_sentences  # 候補(きつい条件)に選ばれた回数

for i in range(num_sentences-1):
    for j in range(i+1, num_sentences):
        sim = sim_matrix[i][j]
        
        if (sim < lower_loose or sim > upper_loose):    # ゆるい条件
            looser_candidates.append([i, j, sim])   
            loose_counts[i] += 1
            loose_counts[j] += 1
        if (sim < lower_strict or sim > upper_strist):    # きつい条件
            strict_candidates.append([i, j, sim])
            strict_counts[i] += 1
            strict_counts[j] += 1

# 結果表示
print(f"\n文章数:{num_sentences}")

# TOP3(きつい条件)
sorted_indices = sorted(enumerate(strict_counts), key=lambda x: x[1], reverse=True)
top3_indices = [index for index, value in sorted_indices[:3]]
print("\n[きつい条件での削除候補 TOP3]")
for i in top3_indices:
    print(f"文章{i+1}:「{sentences[i]}」は削除すべきです！ count:{strict_counts[i]}")
print(f"カウント1:{strict_counts}")

print("\n=== きつい条件の結果 ===")
for i, j, sim in strict_candidates:
    print(f"文章{i+1}:「{sentences[i]}」 と 文章{j+1}:「{sentences[j]}」のどちらかを削除すべきです！")
    print(f"\t類似度:{sim:.3f}")


# TOP3(ゆるい条件)
sorted_indices = sorted(enumerate(loose_counts), key=lambda x: x[1], reverse=True)
top3_indices = [index for index, value in sorted_indices[:3]]
print("\n[ゆるい条件での削除候補 TOP3]")
for i in top3_indices:
    print(f"文章{i+1}:「{sentences[i]}」は削除すべきです！ count:{loose_counts[i]}")
print(f"カウント2:{loose_counts}")

print("=== ゆるい条件の結果 ===")
for i, j, sim in looser_candidates:
    reason = "似過ぎです！" if sim > upper_loose else "遠すぎです！"
    print(f"文章{i+1}:「{sentences[i]}」 と 文章{j+1}:「{sentences[j]}」 は{reason}")
    print(f"\t類似度:{sim:.3f}")



'''
4-4 幻視
5-1 時間混乱
5-2 場所混乱
あまり整理されていない（限界）
'''