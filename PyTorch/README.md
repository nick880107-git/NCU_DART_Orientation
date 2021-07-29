# Pytorch - Multiclass Classfifcation(ToS Race) 

### 1. Training data
資料大小：1000筆(800 for training, 200 for validation)
資料內容：三圍(生命、攻擊、回復)、三圍總和、屬性、種族

![image](https://github.com/nick880107-git/NCU_DART_Orientation/blob/main/PyTorch/image/race_count.png)

### 2. Data Preprocessing
製作Dataset以使用DataLoader，並對資料中的字串進行encoding
- 屬性：因卡片屬性理應無優劣之分，故使用one-hot encoding進行轉換
- 種族：使用lable encoding

### 3. Model Design
根據過往經驗，與其過度堆疊，使用簡易的模型在小訓練集上能有更好的成效
![image](https://github.com/nick880107-git/NCU_DART_Orientation/blob/main/PyTorch/image/model.PNG)
- input_dim : 9 (生命、攻擊、回復、三圍，以及五種屬性的one hot encoding)
- hidden_dim : 180
- output_dim : 8 (8個種族)
- loss function : CrossEntropyLoss (在多分類問題中較常使用的損失函數)
- optimizer : Adam (容易調參且一般情況下就有不錯的成績)

### 4. Leaderboard Score
![image](https://github.com/nick880107-git/NCU_DART_Orientation/blob/main/PyTorch/image/score.PNG)

一開始的分數其實不到5成，猜測是模型設計問題(dropout過多)以及資料處理不熟練(tensor shape及encoding的調整)
歷經幾次調校後才有這樣的分數，不過實際在訓練時計算的accurarcy約落在0.72~0.82(或許是因為沒有固定seed吧)

### 5. Problems during coding
- RuntimeError: Expected object of scalar type Float but got scalar type Long for argument
  - 傳入模型的資料型別有誤，可能是製作dataset時未明確定義型別
  - 解決方法：在dataset定義時，直接宣告tensor的資料型別
- RuntimeError: module must have its parameters and buffers on device cuda:0(device_ids[0]) but found one of them on device:cpu
  - 訓練資料或參數未在同一設備上
  - 解決方法：確認傳入的參數有正確的傳入到指定設備上(包括訓練資料、目標標籤等)
