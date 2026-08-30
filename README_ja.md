# TechJam Conversational E-Commerce Search Challenge

[English README](README.md)

## これは何をする課題か

50,000商品の固定カタログから、顧客が意図している1商品を最大10ターン以内に見つける
会話型商品検索エージェントを作る課題です。

各セッションには、開始前から正解商品が1つ決められています。Agentは顧客役に属性を質問しながら、
毎ターン最大10件の `parent_asin` を順位付きで推薦します。推薦Top 10に正解商品の
`parent_asin` が入ると、そのターンでセッションが終了します。

## 顧客役は何を返すのか

公開ローカル評価器の顧客役はLLMではなく、正解商品のメタデータから作られた
`intent_card` に基づく決定的なシミュレータです。Agentが書いた自然言語の質問を意味解析するのではなく、
構造化された `ask_attribute` を見て返答内容を選びます。

### セッション開始時

顧客役は正解商品のカテゴリとシナリオに応じた英語メッセージを返します。

| シナリオ | 最初の返答 |
| --- | --- |
| `buying` | 商品カテゴリとhard constraintを1つ開示 |
| `browsing` | 商品カテゴリだけを示し、まだ検討中と返答 |
| `intent_override` | 商品カテゴリと、後で撤回する古い希望を提示 |
| `boundary` | `browsing` と同様に曖昧な状態から開始 |

例：

```text
I'm looking for Shoes. A key requirement is: leather.
```

### 2ターン目以降

Agentは次のように、質問したい属性を1つ指定します。

```python
{
    "message": "Do you have a material preference?",
    "ask_attribute": "material",
    "recommendations": [{"parent_asin": "B000..."}]
}
```

利用できる属性は次のとおりです。

```text
category, material, color, size, style, brand,
budget, feature, use_case, other, null
```

顧客役の返答規則は次のとおりです。

- `ask_attribute` と一致する未開示の条件があれば、最大2つ開示する
- 一致する条件がなければ「その属性について追加の希望はない」と返す
- `ask_attribute` が `null` なら、具体的な属性を質問するよう要求する
- `boundary` では最初の具体的な属性質問に対して「希望なし」と返す
- `intent_override` ではターン3または4に、以前の希望を撤回して新しい条件を伝える

例えば `material` を質問して未開示条件に `cotton` があれば、次のような返答になります。

```text
For that, what matters is: cotton.
```

Agentの `message` は顧客向け表示として必要ですが、公開シミュレータがどの情報を開示するかは
`message` ではなく `ask_attribute` で決まります。質問と商品推薦は同じターンに実行できます。

## 正解商品はどこから選ばれるか

「各ターンで50,000商品から顧客役がランダムに1つ選ぶ」という仕組みではありません。
セッションごとに正解商品が事前に固定されています。

公開開発セットでは、[data/public_set.jsonl](data/public_set.jsonl) の各行に次のように入っています。

```json
{
  "sample_id": "public_0001",
  "scenario_type": "buying",
  "user_profile": {"...": "..."},
  "ground_truth": {"parent_asin": "B09PYB7B6Z"}
}
```

公開セットは200セッション・200種類の重複しない正解商品で、すべて50,000商品カタログに存在します。
最終評価では別の800セッションが使われ、`ground_truth`、intent card、シナリオ内部状態は
Agentに渡されません。

ローカル評価器は `ground_truth.parent_asin` でカタログの商品を引き、その商品から次の情報を生成します。

- 商品のカテゴリから最初の大まかな商品カテゴリ
- `features` と `details` からhard constraintとsoft preference
- 商品テキスト中の代表的な素材・色
- 値がある場合は価格条件
- `intent_override` の古い希望、新しい希望、上書きターン

つまり、顧客役の回答は正解商品のメタデータに結び付いています。ただしAgentが正解IDを直接受け取ることはなく、
会話から得た条件でカタログを検索して推定します。

## 商品データはどれか

検索対象はダウンロード後の `data/catalog.jsonl` です。Amazon Reviews 2023の
`Clothing_Shoes_and_Jewelry` から作られた固定50,000商品で、1行が1商品ファミリーです。

| フィールド | 内容 |
| --- | --- |
| `parent_asin` | 商品ファミリーID。推薦と正解判定に使う唯一のキー |
| `title` | 商品名 |
| `features` | 特徴説明の配列 |
| `description` | 商品説明の配列 |
| `price` | 価格。数値、文字列、または `null` |
| `categories` | カテゴリ階層・タグ |
| `details` | 素材、色、サイズ、部門などの構造化属性 |
| `average_rating` | 平均評価 |
| `rating_number` | 評価件数 |
| `store` | ストア・ブランド名。欠損する場合あり |

概略は次の形式です。

```json
{
  "parent_asin": "B000...",
  "title": "...",
  "features": ["..."],
  "description": ["..."],
  "price": 49.99,
  "categories": ["Clothing", "Shoes", "..."],
  "details": {"Department": "...", "Material": "..."},
  "average_rating": 4.4,
  "rating_number": 120,
  "store": "..."
}
```

`public_set.jsonl` は「会話セッションとローカル採点ラベル」、`catalog.jsonl` は
「Agentが検索・推薦する商品集合」という役割分担です。

## 1セッションの処理フロー

```text
public_setの1行から正解parent_asinを読む（評価器のみ）
            ↓
正解商品のメタデータから顧客の条件を生成
            ↓
reset(session_id, user_profile)
            ↓
顧客役が最初のuser_messageを返す
            ↓
Agentが質問属性 + カタログ内のTop 10商品を返す
            ↓
正解があれば終了／なければ顧客役が次の条件を返す
            ↓
最大10ターンまで繰り返す
```

`respond()` に渡されるのは、そのターンの `user_message`、ターン番号、`top_k=10` です。
過去の会話、抽出済み条件、SQL候補集合などは、Agent側が `session_id` ごとに保存します。

## SQL・BERTを使う場合

SQLによる属性フィルタ、BM25、BERT/Sentence Transformerによるsemantic rerankingを組み合わせられます。
例えば次の構成です。

1. `reset()` でセッション状態を作る
2. 顧客の返答から素材、色、価格などを抽出する
3. SQLでhard constraintを満たす候補を絞る
4. BM25やdense retrievalで候補を生成する
5. BERTで候補をrerankし、毎ターンTop 10を返す
6. `intent_override` が来たら古い条件を削除・置換する

早いターンで当てるほどMTTCとEfficiencyが良くなるため、最初の数ターンを質問だけに使うより、
SQL絞り込み中も暫定Top 10を返す方が評価上は有利です。

### 同梱のBERTハイブリッド・ベースライン

現在の `starter/agent.py` は、次の構成をそのまま実行できるベースラインです。

1. 1〜3ターン目は feature、material、use_case を自然な会話で順番に確認する
2. 各回答を会話履歴へ蓄積し、履歴全体をsemantic queryとして使う
3. 各ターンでSQLite FTS5/BM25から250候補を取得する
4. `sentence-transformers/all-MiniLM-L6-v2` の正規化埋め込みで候補をrerankする
5. intent overrideを検出した場合は、最初の商品カテゴリだけを残して古い希望を破棄する

会話中も毎ターン暫定Top 10を返します。初回実行時にモデルを取得し、50,000商品の埋め込みを
`.cache/bert_embeddings/` に保存します。CPUでは数十分かかる場合がありますが、2回目以降は
このキャッシュを再利用します。

```bash
BERT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 BERT_BATCH_SIZE=128 pixi run evaluate
```

ローカルモデルなのでAPIトークン使用量は0です。モデル名とバッチサイズは上記の環境変数で変更できます。
CUDAが利用可能な場合は自動的にCUDAを選択し、利用できない場合はCPUへフォールバックします。
`BERT_DEVICE=cuda` または `BERT_DEVICE=cpu` を指定すれば明示的に上書きできます。
Linux向けPixi環境では `pytorch-gpu` とCUDA 12ランタイム（`linux-64-cuda-12`）を解決します。
NVIDIAドライバが見える環境でインストール後、次で確認できます。

```bash
pixi run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

## セットアップと実行

[Pixiをインストール](https://pixi.sh/latest/installation/)し、リポジトリルートで実行します。

```bash
pixi install
pixi run download-data
pixi run data-info
pixi run check
pixi run evaluate
```

`pixi run download-data` は公式Releaseからカタログを取得し、SHA-256を検証して
`data/catalog.jsonl` に展開します。ファイルは大きいためGit管理対象外です。

主なPixiタスク：

| タスク | 内容 |
| --- | --- |
| `pixi run download-data` | 商品カタログをダウンロード・検証・展開 |
| `pixi run data-info` | 行数、シナリオ数、フィールド型を表示 |
| `pixi run validate-data` | 公開セットとカタログを検証 |
| `pixi run test` | ユニットテストを実行 |
| `pixi run check` | テストと全データ検証を実行 |
| `pixi run evaluate` | 公開200セッションでAgentを評価 |

## 評価指標

- Hit Rate@10: 10ターン以内に正解したセッションの割合
- MRR: 最初に正解したターンにおける正解商品の順位の逆数平均
- MTTC: 最初に正解したターン。失敗は11ターンとして計算
- Efficiency: `(11 - MTTC) / 10` を0〜1に制限

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

## 関連ファイル

| ファイル | 内容 |
| --- | --- |
| [starter/agent.py](starter/agent.py) | SQLite FTS5/BM25を使う編集可能なベースライン |
| [evaluator/local_evaluator.py](evaluator/local_evaluator.py) | 顧客シミュレータと採点処理 |
| [docs/competition_specification.md](docs/competition_specification.md) | 競技仕様 |
| [docs/agent_api_contract.json](docs/agent_api_contract.json) | Agent APIの機械可読スキーマ |
| [docs/evaluation_config.json](docs/evaluation_config.json) | ターン数、Top K、評価式 |
| [data/README.md](data/README.md) | データフィールドと取得方法 |
