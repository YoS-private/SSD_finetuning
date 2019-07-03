import json
import matplotlib.pyplot as plt
import copy
import chainer
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
import os
from chainercv.utils import read_image

from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainer.iterators import serial_iterator

import chainermn

from chainercv import utils
from chainercv import transforms
from chainercv.visualizations import vis_bbox
from chainercv.chainer_experimental.datasets.sliceable import ConcatenatedDataset
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import resize_with_random_interpolation
from chainercv.links.model.ssd import random_distort
from chainercv.links import SSD300
from chainercv.links import SSD512

from chainercv.links import SSD300
import numpy as np

# 学習データをインポートするクラス
# GetterDatasetを継承
class originalDataset(GetterDataset):
    # 初期値はとりあえずはsplit='train'
    def __init__(self,split='train',use_difficult=False,return_difficult=False):
        super(originalDataset, self).__init__()
        data_dir = 'ssd_picts/'+split+'/'

        file_names = []
        for file in os.listdir(data_dir+'image'):
            file_names.append(file)

        self.filenames = file_names
        self.data_dir = data_dir
        self.use_difficult = 0 # difficultラベルがないので

        # _get_imageと_get_annotationsで画像とそのアノテーションをインポート
        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        # difficultをリターンする必要がなければdifficultは入れる必要ないので
        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

    # ファイル数を出力
    def __len__(self):
        return len(self.filenames)

    # 画像のインポート
    def _get_image(self,i):
        file_name = self.filenames[i]
        img = read_image(self.data_dir+'image/'+file_name)
        return img

    # i番目の画像のアノテーション情報(メタデータ)をインポート
    def _get_annotations(self,i):
        bbox = np.empty((0,4), float)
        label = np.empty((0,1), int)
        difficult = []
        filename = self.filenames[i]
        # メタデータから該当データ探索
        f = open(self.data_dir+'metadata.json')
        json_data = json.load(f)['_via_img_metadata']
        tmp_picts = list(json_data.values())
        tmp_data = {}
        objs = [p['regions'] for p in tmp_picts if p['filename'] == filename][0]

        # 領域それぞれに対して領域の角の点を抽出，ラベルも同時に定義
        for obj in objs:
            xmax = int(obj['shape_attributes']['x']) + int(obj['shape_attributes']['width'])
            ymax = int(obj['shape_attributes']['y']) + int(obj['shape_attributes']['height'])
            tmp_bbox=np.array([int(obj['shape_attributes']['y']), int(obj['shape_attributes']['x']), ymax, xmax])
            tmp_label = np.array([int(obj['region_attributes']['label'])])
            bbox = np.append(bbox,np.array([tmp_bbox]), axis = 0)
            label = np.append(label, np.array([tmp_label]), axis = 0)
        difficult.append(False)
        bbox = np.array(bbox,dtype=np.float32)
        difficult = np.array(difficult,dtype=np.bool) # 今回は使わない
        label = np.array(label,dtype=np.int32)
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        return bbox, label, difficult


# 学習済みデータセットとクラス数が異なる時にスキップする重みを指定する関数
# 入力はモデル(source,destination)
def get_shape_mismatch_names(src, dst):
    mismatch_names = []
    src_params = {p[0]: p[1] for p in src.namedparams()}
    # クラス数が違うところだけをmismatch_namesにappend→出力
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        dst_param = dst_named_param[1]
        src_param = src_params[name]
        if src_param.shape != dst_param.shape:
            mismatch_names.append(name)
    return mismatch_names


# SSDのモデルの定義
class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def forward(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss

# 学習データの水増しとSSDに入力するための準備処理
class Transform(object):
    def __init__(self, coder, size, mean):
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # 5段階のステップでデータの水増しを行う
        # 1. 色の拡張
        # 2. ランダムな拡大
        # 3. ランダムなトリミング
        # 4. ランダムな補完の再補正
        # 5. ランダムな水平反転


        img, bbox, label = in_data

        # 1. 色の拡張
        # 明るさ，コントラスト，彩度，色相を組み合わせ，データ拡張をする
        img = random_distort(img)

        # 2. ランダムな拡大
        if np.random.randint(2):
            # キャンバスの様々な座標に入力画像を置いて，様々な比率の画像を生成し，bounding boxを更新
            img, param = transforms.random_expand(img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. ランダムなトリミング
        img, param = random_crop_with_bbox_constraints(img, bbox, return_param=True)
        # トリミングされた画像内にbounding boxが入るように調整
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'],
            x_slice=param['x_slice'],
            allow_outside_center=False,
            return_param=True)
        label = label[param['index']]

        # 4. ランダムな補完の再補正
        ## 画像とbounding boxのリサイズ
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. ランダムな水平反転
        ## 画像とbounding boxをランダムに水平方向に反転
        img, params = transforms.random_flip(img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(bbox, (self.size, self.size), x_flip=params['x_flip'])

        # SSDのネットワークに入力するための準備の処理
        img -= self.mean
        ## SSDに入力するためのloc(デフォルトbounding boxのオフセットとスケール)と
        ## mb_label(クラスを表す配列)を出力
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label







# 学習データ，評価データ，テストデータの属性を読み込み
# 引数にはその属性があるパスを入力
# train_dataset = outDataset('./ssd_picts/train/train.json')
# valid_dataset = outDataset('./ssd_picts/valid/valid.json')
# test_dataset = outDataset('./ssd_picts/test/test.json')

# chainerCVでSSDを動かす通常の動作
## モデルの定義(重みは PASCAL VOC 2007と2012の学習済みデータで実行)
### 表示するためのスコアの閾値は0.6，non_maximun_suppressionの閾値は0.45
model = SSD300(n_fg_class=len(voc_bbox_label_names), pretrained_model='voc0712')
model.score_thresh = 0.50
model.nms_thresh = 0.45
img = utils.read_image('ssd_picts/test/image/000000581155.jpg', color=True)
# モデルを使って予測
bboxes, labels, scores = model.predict([img])
bbox, label, score = bboxes[0], labels[0], scores[0]

vis_bbox(img, bbox, label, score, label_names=voc_bbox_label_names)
# 表示
# plt.show()


# fine-tuningの準備
# pre-trainedモデルの重みの一部を使う．
src = SSD300(pretrained_model='voc0712')
# voc0712: 50種類
# dst(fine-tuning結果のモデル)は21クラスに設定
# dstとsrcでクラス数が異なる
dst = SSD300(n_fg_class=21)
# 重みをとりあえず全ての層において初期化
dst(np.zeros((1, 3, dst.insize, dst.insize), dtype=np.float32))

# ignore_names以外の層に関しては，出力先のdstにパラメータを出力
ignore_names = get_shape_mismatch_names(src, dst)
src_params = {p[0]: p[1] for p in src.namedparams()}
for dst_named_param in dst.namedparams():
    name = dst_named_param[0]
    if name not in ignore_names:
        dst_named_param[1].array[:] = src_params[name].array[:]

# 読み込まれているかチェック．
# 読み込まれていない場合のみ出力としてAssertionエラーがでる．
np.testing.assert_equal(dst.extractor.conv1_1.W.data,
                        src.extractor.conv1_1.W.data)
# スキップした層の名前を出力
print(ignore_names)




# 学習
## 変数定義
gpu = 0 # gpuのID
batchsize = 16 # バッチサイズ
test_batchsize = 8
iteration = 120
step = [80, 100]
out = 'result' # 出力ファイルパス
resume = None # 重み

label = ['man','woman']

# dstのignore_namesを学習する
model = dst

# score_threshとnms_threshの値を変更
model.use_preset('evaluate')
train_chain = MultiboxTrainChain(model)

chainer.cuda.get_device_from_id(gpu).use()
model.to_gpu()

# 学習データの取り込み with 水増し
train = TransformDataset(
    originalDataset(split='train'),
    ('img', 'mb_loc', 'mb_label'),
    Transform(model.coder, model.insize, model.mean))
train_iter = chainer.iterators.MultiprocessIterator(train, batchsize)

# テストデータの取り込み
test = originalDataset(
    split='test', use_difficult=True, return_difficult=True)
test_iter = chainer.iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)

# initial lr is set to 1e-3 by ExponentialShift
# 確率的勾配法
# setupで初期化
optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(train_chain)

# 線形関数：y=Wx+b(Wは重み，bはバイアス)
# bの時：勾配のスケールを2倍にあげる．
# Wの時：重みの減衰率を0.0005倍にする．
# フック：プログラム中の特定の箇所に独自の処理を追加できるようにする，
for param in train_chain.params():
  if param.name == 'b':
    param.update_rule.add_hook(GradientScaling(2))
  else:
    param.update_rule.add_hook(WeightDecay(0.0005))

# 最適化関数とトレーニングデータセットを入力
# 学習部分ににoptimizerを繋げる．
updater = training.updaters.StandardUpdater(
    train_iter, optimizer, device=gpu)
trainer = training.Trainer(
    updater, (iteration, 'iteration'), out)
# 指数関数的に学習率を変更できるように設定
trainer.extend(
    extensions.ExponentialShift('lr', 0.1, init=1e-3),
    trigger=triggers.ManualScheduleTrigger(step, 'iteration'))

# 学習とともにAPやmAPを評価するように設定
trainer.extend(
    DetectionVOCEvaluator(
        test_iter, model, use_07_metric=True,
        label_names=(voc_bbox_label_names+('new_label',))))

log_interval = 10, 'iteration'
# 評価結果の出力
trainer.extend(extensions.LogReport(trigger=log_interval))
trainer.extend(extensions.observe_lr(), trigger=log_interval)
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'lr',
     'main/loss', 'main/loss/loc', 'main/loss/conf',
     'validation/main/map']),
               trigger=log_interval)
trainer.extend(extensions.ProgressBar(update_interval=10))

# 結果の表示
trainer.extend(
    extensions.snapshot(),
    trigger=triggers.ManualScheduleTrigger(
        step + [iteration], 'iteration'))
trainer.extend(
    extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
    trigger=(iteration, 'iteration'))

if resume:
  serializers.load_npz(resume, trainer)

trainer.run()
