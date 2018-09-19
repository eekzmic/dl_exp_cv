- train_mnist_mlp.py
MNISTデータセットをMLPによって学習します
MLPの定義はnet.pyのMLPクラスにあります

- test_mnist_mlp.py
手書き文字を読み込んで学習済みモデルに通し，その数字が何であるかを予測します

- train_mnist_cnn.py
MNISTデータセットをCNNによって学習します
CNNの定義はnet.pyのMnistCNNクラスを完成させてください

- train_cifar10.py
cifar10をCNNで学習します．
CNNの定義はnet.pyのCifarCNNにあります．
データセットの読み込みの定義をdataset.pyに書いて完成させてください

- test_cifar10.py
cifar10のテスト用データセットを読み込んでaccuracyを出力します

- create_db.py
cifar10の訓練用データセットをVGGに通して深層特徴とパスのリストをdbディレクトリに保存します

- search.py
create_dbで作った深層特徴リストと，入力した画像の深層が最も近いもののtop kのパスを出力します
