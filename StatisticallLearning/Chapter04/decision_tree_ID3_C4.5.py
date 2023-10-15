# 使用决策树来实现ID3和C4.5算法

import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_digits


class DecisionTree:
    def __init__(self, data, labels, sort, max_depth):
        self.data = data
        self.labels = labels
        self.sort = sort
        self.max_depth = max_depth
        self.root = None

    # 生成树模型
    def create_tree(self, data, labels, sort, depth):
        # 判断是否到达树的末尾
        tree = {'over': False}
        # 如果当前结点的中的数据个数为0，或者当前结点中的标签只有一个种类,或者当前类别中都是相同的类别，那么此时也终止判断
        if len(data) == 0 or len(np.unique(labels)) == 1 or self.affirm_sort(labels) or depth > self.max_depth:
            # 此时表示已经到达了树的构建的末尾
            tree['over'] = True
            # 这个时候，使用表决的办法得到最后的返回种类值
            tree['label'] = self.vote_label(labels)
            return tree

            # 计算出信息熵
        entropy = self.cal_entropy(labels)

        # 用来存放当前的信息增益/信息增益比
        gains = []
        gains_ratio = []
        num_features = data.shape[1]
        assert sort in ['ID3', 'C4.5'], '请输入正确的类别，ID3或者C4.5'
        if sort in ['ID3', 'C4.5']:
            # 这里计算信息增益是ID3和C4.5算法都需要计算的
            # 循环遍历每个特征，选出最佳的特征
            for num_feature in range(num_features):
                # 得到当前特征中的类别数以及每个类别对应的个数
                unique_sorts, sort_counts = np.unique(data[:, num_feature], return_counts=True)
                # 计算每个类别的条件熵
                for unique_sort in unique_sorts:
                    current_index = data[:, num_feature] == unique_sort
                    # 计算当前类别的条件熵
                    current_labels_entropy = self.cal_entropy(labels[current_index])
                    # 得到当前特征的中的信息增益
                    entropy += current_labels_entropy
                    # 将信息增益添加进去
                gains.append(entropy)
                # 如果当前输入的算法是C4.5,那么就继续计算信息增益比
                if sort == 'C4.5':
                    # 将当前特征传入进去，就可以得到每个特征对应的类别的总的信息量
                    feature_entropy = self.cal_entropy(data[:, num_feature])
                    # 计算得到信息增益比
                    entropy_ratio = entropy / feature_entropy
                    # 将信息增益比添加进去
                    gains_ratio.append(entropy_ratio)
        # 取出最大的信息增益或者信息增益比，然后根据信息增益或者信息增益比最大位置的索引得到对应的划分特征
        tree['child_tree'] = {}
        tree['best_feature'] = {}
        if sort == 'ID3':
            # 得到最佳划分特征的坐标
            max_feature_index = np.argmax(gains)
        else:
            # 如果类型是C4.5的话，那么需要根据信息增益比来进行划分
            # 首先得到平均信息增益
            mean_gain = np.mean(gains)
            max_feature_index = -1
            max_gain_ratio = -1
            for i, (gain, gain_ratio) in enumerate(zip(gains, gains_ratio)):
                # 如果当前位置的信息增益大于平均信息增益，并且该信息增益比是最大的话
                if gain > mean_gain and gain_ratio > max_gain_ratio:
                    max_feature_index = i
                    max_gain_ratio = gain_ratio
        # 将最佳划分特征存放起来，将来好对输入的测试模型进行特征点的切分
        tree['best_feature'] = max_feature_index
        # 此时已经得到了最佳划分特征，然后可以根据这个最佳划分特征对数据进行划分
        # 分别得到当前特征的种类数，以及每个种类对应的样本数
        split_sorts, split_counts = np.unique(data[:, max_feature_index], return_counts=True)
        # 然后根据当前特征结点构建
        for (split_sort, split_count) in zip(split_sorts, split_counts):
            data_sort_index = data[:, max_feature_index] == split_sort
            # 将当前的最佳特征进行剔除,这里一定要对data进行破坏
            data_split = np.hstack((data[:, :max_feature_index], data[:, max_feature_index + 1:]))
            # 然后进行循环构建树
            tree['child_tree'][split_sort] = self.create_tree(data_split[data_sort_index, :],
                                                              labels[data_sort_index],
                                                              sort, depth + 1)
        # 如果一直到树构建的最后，那么也开始退出树的构建
        return tree

    # 定义计算信息熵的函数
    def cal_entropy(self, labels):
        # 返回当前标签中每个类别对应的数量
        _, counts = np.unique(labels, return_counts=True)
        ratio = counts / labels.shape[0]
        # 利用对应元素相乘的方式来计算得到信息熵
        entropy = (- ratio * np.log2(ratio)).sum()
        return entropy

    # 判断当前标签中是否都是相同的结果。如果是的话，那么就没有必要生成下去的，此时应该停止树模型的创建
    def affirm_sort(self, labels):
        if len(np.unique(labels)) == 1:
            return True
        return False

    # 采用表决的思想得到最后的标签值
    def vote_label(self, labels):
        label, count = np.unique(labels, return_counts=True)
        return label[np.argmax(count)]

    # 对使用构建好的树模型进行预测
    def use_decision_tree_predict(self, root, test_data):
        # 这里每次只传入一个数据，对一个数据进行决策
        if root['over']:
            return root['label']
        current_feature = root['best_feature']
        # 重新对输入的数据进行拼接
        test_data = test_data.reshape(1, -1)
        test_spilt = np.hstack((test_data[:, :current_feature], test_data[:, current_feature + 1:]))
        # 根据当前测试数据的该最佳特征处的值，判断应该进行树的哪一边
        return self.use_decision_tree_predict(root['child_tree'][test_data[:, current_feature][0]], test_spilt)

    # 进行测试
    def test(self, root, test_data, test_label):
        # 每次传入一个样本的数据
        acc = 0
        for i in tqdm(range(test_data.shape[0])):
            prediction = self.use_decision_tree_predict(root, test_data[i, :])
            if prediction == test_label[i]:
                acc += 1
        # 得到最后预测的准确率
        return acc / test_data.shape[0]


if __name__ == '__main__':
    digits = load_digits()
    data = digits.data
    labels = digits.target
    labels = (labels > 4).astype(int)  # 0-4 设为标签0 5-9 设为标签1
    data = (data > 7).astype(int)
    shuffle_index = np.random.permutation(data.shape[0])
    digits = data[shuffle_index]
    labels = labels[shuffle_index]  # 1797

    # 选择前面1200个数据作为训练数据，后面的数据座位测试数据
    trains_data = data[:1200, :]
    trains_labels = labels[:1200]

    tests_data = data[1200:, :]
    tests_labels = labels[1200:]
    np.random.seed(42)
    decision_tree = DecisionTree(trains_data, trains_labels, 'ID3', 5)
    decision_tree.root = decision_tree.create_tree(trains_data, trains_labels, 'ID3', 0)
    evaluation_accuracy = decision_tree.test(decision_tree.root, tests_data, tests_labels)
    print(evaluation_accuracy)
