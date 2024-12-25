

# Django project for News Sentiment Indicator
This project leverages natural language models to perform sentiment analysis on industry news content and calculate a sentiment index. Using the Django framework, an intuitive and interactive analytics interface was developed, allowing users to flexibly adjust modeling parameters and visualize sentiment index results. This empowers users to accurately interpret sentiment trends and make data-driven decisions.

# 情緒指標建立
本分析採用字典法，適用於各種中文資料來源的情緒計算，主要情緒計算的方式參考自 Barbaglia (2024)，並採用兩套中文情緒字典，分別建構自 Bian (2019) 以及 Du (2021)，並將簡體中文轉換為繁體中文。本程式使用字典法來示範建立每日新聞情緒指標，根據不同計算邏輯，我們將分別定以下五種不同計算方式的情緒指標。而在定義這些指標之前，我們先定義相關的符號。

假設每日新聞的總數為 $N_t$，其中 $t$ 表示日期時間戳記（例如 2024 年 1 月 1 日）。對於當天的每一篇新聞，標記為 $A_{t,j}$，其中 $j = 1, \dots, N_t$。每篇新聞 $A_{t,j}$ 首先經由 CKIP 工具進行斷詞（Word Segmentation）和詞性分析（POS Tagging），以得到更精細的詞彙單位。每篇新聞 $A_{t,j}$ 可表示為一組斷詞結果 $\{ WS_{t,ji} \mid i = 1, \dots, M_{t,j} \}$，其中 $M_{t,j}$ 表示在時間點 $t$ 下，新聞 $A_{t,j}$ 的總斷詞數。

為了容易理解本範例情緒分數計算的邏輯，我們將字典來自於 Bian (2019) 以及 Du (2021) 分別定義為字典 $A$ 與 $B$，並分別延伸兩本字典的交集與聯集，分別定義為 $A \cap B$ 以及 $A \cup B$。而本範例所關心的關鍵字亦包含三種組合：產業、總體經濟與不分類，分別標注為 $\mathbb{K}_{\mathrm{ind}}$、$\mathbb{K}_{\mathrm{macro}}$ 與 $\mathbb{K}$。

## 情緒指標一：

第一個指標，以情緒詞彙集合字典 $A$ ($\mathrm{DICT}_{A,\mathrm{ps}}$) 為例，每日情緒分數的計算基於正面情緒詞彙集合 $\mathrm{DICT}_{A,\mathrm{ps}}$ 中字詞的出現頻率。具體而言，對於每篇新聞 $A_{t,j}$，正面情緒分數 $S_{t,j}^{\mathrm{A,ps}}$ 定義為新聞中所有斷詞 $WS_{t,ji}$ 屬於正面情緒字典 $\mathrm{DICT}_{A,\mathrm{ps}}$ 的詞彙總數：

$$
S_{t,j}^{\mathrm{A,ps}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{ps}} \}
$$

其中，指示函數 $\mathbf{1}\{ \cdot \}$ 當 $WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{ps}}$ 時取值為 1，否則取值為 0。以同樣的方式亦可以根據負面情緒詞彙集合 $\mathrm{DICT}_{A,\mathrm{neg}}$ 計算：

$$
S_{t,j}^{\mathrm{A,neg}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{neg}} \}
$$

最後我們可以定義時間點 $t$ 下的情緒分數為：

$$
S_{t}^{\mathrm{A}} = \sum_{j=1}^{N_t} \left( S_{t,j}^{\mathrm{A,ps}} - S_{t,j}^{\mathrm{A,neg}} \right)。
$$

## 情緒指標二：

相較於情緒指標一針對每一天的所有新聞計算情緒分數，本範例透過關鍵字字典篩選每日新聞，僅將包含產業、總體經濟或不分類關鍵字的新聞納入計算。對於每日新聞 $A_{t,j}$，若其斷詞結果包含至少一個產業關鍵字（即 $\exists i$ 使得 $WS_{t,ji} \in \mathbb{K}_{\mathrm{ind}}$），則計算該篇新聞的正面與負面情緒分數：

$$
S_{t,j}^{\mathrm{A,ps,ind}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{ps}} \} \cdot \mathbf{1}\left\{ \exists k \in \{1, \dots, M_{t,j}\} \text{ such that } WS_{t,jk} \in \mathbb{K}_{\mathrm{ind}} \right\}
$$

```latex
S_{t,j}^{\mathrm{A,ps,ind}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{ps}} \} \cdot \mathbf{1}\left\{ \exists k \in \{1, \dots, M_{t,j}\} \text{ such that } WS_{t,jk} \in \mathbb{K}_{\mathrm{ind}} \right\}
```
$$
S_{t,j}^{\mathrm{A,neg,ind}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{neg}} \} \cdot \mathbf{1}\left\{ \exists k \in \{1, \dots, M_{t,j}\} \text{ such that } WS_{t,jk} \in \mathbb{K}_{\mathrm{ind}} \right\}
$$
```latex
S_{t,j}^{\mathrm{A,neg,ind}} = \sum_{i=1}^{M_{t,j}} \mathbf{1}\{ WS_{t,ji} \in \mathrm{DICT}_{A,\mathrm{neg}} \} \cdot \mathbf{1}\left\{ \exists k \in \{1, \dots, M_{t,j}\} \text{ such that } WS_{t,jk} \in \mathbb{K}_{\mathrm{ind}} \right\}
```

時間點 $t$ 下，產業相關的情緒分數定義為：

$$
S_{t}^{\mathrm{A,ind}} = \sum_{j=1}^{N_t} \left( S_{t,j}^{\mathrm{A,ps,ind}} - S_{t,j}^{\mathrm{A,neg,ind}} \right)
$$

## 情緒指標三：

情緒指標一與二僅基於情緒詞彙的出現次數進行分數計算，未考慮新聞篇幅影響。本範例提出標準化情緒分數的方法，按每篇新聞的斷詞總數 $M_{t,j}$ 進行標準化：

$$
\tilde{S}_{t}^{\mathrm{A}} = \sum_{j=1}^{N_t} \frac{\left( S_{t,j}^{\mathrm{A,ps}} - S_{t,j}^{\mathrm{A,neg}} \right)}{M_{t,j}}
$$

## 情緒指標四：

情緒指標四僅分析包含關鍵字的句子，並計算該句的情緒分數。定義關鍵字所在的句子為 $Sen_{t,jl}$，計算正面與負面情緒分數為：

$$
\bar{S}_{t,j}^{\mathrm{A,ps}} = \sum_{l} \sum_{i \in Sen_{t,jl}} \mathbf{1}\{ i \in \mathrm{DICT}_{A,\mathrm{ps}} \}
$$

$$
\bar{S}_{t,j}^{\mathrm{A,neg}} = \sum_{l} \sum_{i \in Sen_{t,jl}} \mathbf{1}\{ i \in \mathrm{DICT}_{A,\mathrm{neg}} \}
$$

## 情緒指標五：

基於情緒指標四，按句子的斷詞數進行標準化：

$$
\tilde{\bar{S}}_{t,j}^{\mathrm{A,ps}} = \sum_{l} \frac{\sum_{i \in Sen_{t,jl}} \mathbf{1}\{ i \in \mathrm{DICT}_{A,\mathrm{ps}} \}}{|Sen_{t,jl}|}
$$

$$
\tilde{\bar{S}}_{t,j}^{\mathrm{A,neg}} = \sum_{l} \frac{\sum_{i \in Sen_{t,jl}} \mathbf{1}\{ i \in \mathrm{DICT}_{A,\mathrm{neg}} \}}{|Sen_{t,jl}|}
$$

### 表格：情緒指標定義

| 設定                | 字典 A               | 字典 B               | 兩本交集                | 兩本聯集                | 備註            |
|---------------------|----------------------|----------------------|-------------------------|-------------------------|-----------------|
| 標準化             | $\tilde{S}_{t}^{\mathrm{A}}$ | $\tilde{S}_{t}^{\mathrm{B}}$ | $\tilde{S}_{t}^{\mathrm{A\cap B}}$ | $\tilde{S}_{t}^{\mathrm{A\cup B}}$ | 標準化數據 |
| 沒標準化           | $S_{t}^{\mathrm{A}}$ | $S_{t}^{\mathrm{B}}$ | $S_{t}^{\mathrm{A\cap B}}$ | $S_{t}^{\mathrm{A\cup B}}$ | 原始數據     |
| 產業關鍵字         | $\tilde{S}_{t}^{\mathrm{A,ind}}$ | $\tilde{S}_{t}^{\mathrm{B,ind}}$ | $\tilde{S}_{t}^{\mathrm{A\cap B,ind}}$ | $\tilde{S}_{t}^{\mathrm{A\cup B,ind}}$ | 標準化數據 |
| 總體關鍵字         | $\tilde{S}_{t}^{\mathrm{A,macro}}$ | $\tilde{S}_{t}^{\mathrm{B,macro}}$ | $\tilde{S}_{t}^{\mathrm{A\cap B,macro}}$ | $\tilde{S}_{t}^{\mathrm{A\cup B,macro}}$ | 標準化數據 |
| 所有關鍵字         | $\tilde{S}_{t}^{\mathrm{A,all}}$ | $\tilde{S}_{t}^{\mathrm{B,all}}$ | $\tilde{S}_{t}^{\mathrm{A\cap B,all}}$ | $\tilde{S}_{t}^{\mathrm{A\cup B,all}}$ | 標準化數據 |
