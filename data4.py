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
