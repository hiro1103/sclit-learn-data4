import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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
