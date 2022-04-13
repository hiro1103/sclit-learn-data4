import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sympy import rotations, subsets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# サンプルデータを作成
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''

# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))


# 各特徴量の欠測値をカウント
df.isnull().sum()

# 欠測値を含む行を削除
df.dropna()

# 欠測値を含む列を削除
df.dropna(axis=1)

# すべての列がNaNである行だけを削除
df.dropna(how='all')

# 非NaN値が4つ未満の行を削除
df.dropna(thresh=4)

# 特定の列にNaNが含まれている行だけを削除
df.dropna(subset=['C'])

# 欠測値補完のインスタンスを生成（平均値補完）
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# データを適合させる
imr = imr.fit(df.values)
# 補完を実行
imputed_data = imr.transform(df.values)

df.fillna(df.mean())

# サンプルデータを生成（Tシャツの色、サイズ、価格、クラスラベル）
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 13.5, 'class2'],
])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
df

# Tシャツのサイズと整数を対応させる辞書を作成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
df

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# クラスラベルと整数を対応させる辞書を作成
class_mapping = {label: idx for idx,
                 label in enumerate(np.unique(df['classlabel']))}
class_mapping

# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
df

# 整数とクラスラベルを対応させる辞書を作成
inv_class_mapping = {v: k for k, v in class_mapping.items()}

# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df

# ラベルエンコーダーのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
y

# クラスラベルを文字列に戻す
class_le.inverse_transform(y)

# Tシャツの色、サイズ、価格を抽出
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

X = df[['color', 'size', 'price']].values
# One-hotエンコーダーの生成
color_ohe = OneHotEncoder()
# One-hotエンコーディングを実行
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),
                              ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)

# One-hotエンコーディングを実行
pd.get_dummies(df[['price', 'color', 'size']])

# One-hotエンコーディングを実行
pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)

# One-hotエンコーダの生成
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([('onehot', color_ohe, [0]),
                              ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)

# wineデータセットを読み込む
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header=None
)

# 列名を設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Megesinu', 'Total phenols', 'Flavanoids',
                   'Monflavanoid phenols', 'Proanthcyannis', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))

# wineデータセットの先頭5行を表示
print(df_wine.head())

# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割(30%をテストデータにする)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
# 訓練データをスケーリング
X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
X_test_norm = mms.transform(X_test)

ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex-ex.mean())/ex.std())
print('normalized:', (ex-ex.min())/(ex.max()-ex.min()))
# 標準化のインスタンスを生成(平均0,標準偏差1に変換)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')
# L1正規化ロジスティク回帰のインスタンスを生成：逆正則化パラメータC=1.0はデフォルト値であり、
# 値を大きくしたり小さくしたりすると、正則化の効果を強めたり弱めたりできる。
lr = LogisticRegression(penalty='l1', C=1.0,
                        solver='liblinear', multi_class='ovr')
# 訓練データに適合
lr.fit(X_train_std, y_train)
# 訓練データに対する正解率を算出
print('Training accuracy:', lr.score(X_train_std, y_train))
# テストデータに対する正解率を算出
print('Test accuracy:', lr.score(X_test_std, y_test))

print(lr.intercept_)
# 重みの係数を表示
print(lr.coef_)

# 描画の準備
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
# 各係数の色のリスト
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
# 空のリストを生成（重み係数、逆正則化パラメータ）
weights, params = [], []
# 逆正則化パラメータの値ごとに処理
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear',
                            multi_class='ovr', random_state=0)

    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をNumpy配列に変換
weights = np.array(weights)
# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ、縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1],
             color=color)

# y=0に黒い波線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
# 横軸の範囲の設定
plt.xlim([10**(-5), 10**5])
# 軸のラベルの設定
plt.ylabel('weight coefficient')
plt.xlabel('C')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(
    1.38, 1.03), ncol=1, fancybox=True)
plt.show()


class SBS():

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1) -> None:

        # 特徴量を評価する指標
        self.scoring = scoring
        # 推定器
        self.estimator = estimator
        # 選択する特徴量の個数
        self.k_featurs = k_features
        # テストデータの割合
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # すべての特徴量の個数、列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # すべての特徴量を用いてスコアを算出
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)

        self.scores_ = [score]
        # 特徴量が指定した個数になるまで処理を繰り返す
        while dim > self.k_featurs:
            # 空のスコアリストを作成
            scores = []
            # 空の列インデックスリストを作成
            subsets = []
            # 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復
            for p in combinations(self.indices_, r=dim-1):
                # スコアを算出して格納
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)

            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)
            # 最良のスコアとなる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            # 特徴量の個数を1つだけ減らして次のステップへ
            dim -= 1
            # スコアを格納
            self.scores_.append(scores[best])
        # 最後に格納したスコア
        self.k_featurs = self.scores_[-1]
        return self

    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルを適合
        self.estimator.fit(X_train[:, indices], y_train)
        # テストデータを用いてクラスラベルを予測
        y_pred = self.estimator.predict(X_test[:, indices])
        # 真のクラスラベルと予測値を用いてスコアを算出
        score = self.scoring(y_test, y_pred)
        return score


# K最近傍法分類器のインスタンスを生成
knn = KNeighborsClassifier(n_neighbors=5)
# 逐次後退選択のインスタンスを生成（特徴量の個数が1になるまで特徴量を選択)
sbs = SBS(knn, k_features=1)
# 遂次後退選択を実行
sbs.fit(X_train_std, y_train)

# 特徴量の個数のリスト(13,12,.........1)
k_feat = [len(k) for k in sbs.subsets_]
# 横軸を特徴量の個数、縦軸をスコアとした折れ線グラフのプロット
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# 13個すべての特徴量を用いてモデルを適合
knn.fit(X_train_std, y_train)
# 訓練の正解率を出力
print('Training accuracy:', knn.score(X_train_std, y_train))

# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std, y_test))

# 3つの特徴量を用いてモデルを適合
knn.fit(X_train_std[:, k3], y_train)
# 訓練の正解率を出力
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))

# wineデータセットの特徴量の名称
feat_labels = df_wine.columns[1:]
# ランダムフォレストオブジェクトの生成（決定木の個数:500）
forest = RandomForestClassifier(n_estimators=500, random_state=1)
# モデルを適合
forest.fit(X_train, y_train)
# 特徴量の重要度を抽出
importances = forest.feature_importances_
# 重要度の降順で特徴量のインデックスを抽出
indices = np.argsort(importances)[::-1]
# 重要度の降順で特徴量の名称、重要度を表示
for f in range(X_train.shape[1]):
    print("(%2d) %-*s %f" %
          (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
