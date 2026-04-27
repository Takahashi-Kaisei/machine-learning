# machine-learning

機械学習の学習用リポジトリです。Jupyter Notebook を中心に、理論メモ、発表資料、サンプルコードをまとめています。

## Directory Layout

- `notebook/`: 学習・検証用の Notebook
- `sample/`: 演習用やサンプルの Notebook
- `talk/`: 発表用に切り出した Notebook
- `markdown/`: 理論や実装の文章メモ
- `mindmap/`: 章立てや論点整理
- `dataset/`: ローカルで使うデータセット
- `pdf/`: 生成済みの PDF 資料
- `materials/`: Slidev ベースのスライド資料

## Python Environment

Python は `3.13` を前提にしています。

```bash
uv sync
uv run python main.py
uv run jupyter notebook
```

Notebook から再利用する処理が増えたら、ルート配下または `src/` 配下に Python モジュールとして切り出してください。

## Slides

`materials/` は独立した Slidev プロジェクトです。

```bash
cd materials
npm install
npm run dev
npm run build
```

## Repository Rules

- 依存物やローカル生成物はコミットしない
- Notebook はトピック単位で分ける
- 説明文は `markdown/`、図解や構造化メモは `mindmap/` に置く
- 一時的な検証コードを増やしすぎる前に整理方針を README に反映する
