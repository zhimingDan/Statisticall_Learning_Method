# 统计学习算法第三章：K近邻的实现
# k近邻：就是利用相同类别的事物的特征是相似的，是会聚拢在一块的

import heapq
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import datasets


# 一个节点类
class Node(object):
    def __init__(self, data, left_node, right_node, dim):
        # 对于二叉树中一个树的节点有哪些变量？
        # 首先:这个节点肯定需要存储数据
        # 其次:这个节点肯定有左右两个子节点(如果没有的话，那么这个两个子节点为None)
        # 最后:我们需要有节点的深度 -> 因为在对数据进行判断的时候，我们需要根据深度来构建一个kd树
        self.data = data
        self.left_node = left_node
        self.right_node = right_node
        self.dim = dim


# 有了节点类，我们怎么构建kd树呢？
# 我希望：当我创建一个KDTree类的实例的时候，就能自动创建好kd树
class KDTree(object):
    def __init__(self, data):
        # 因为传入的数据是data和labels进行整合后的数据，因此需要对特征数-1
        self.root = self.create_kd_tree(data, 0, data.shape[1] - 1)

    # 创建一个kd树
    # 这里为了方便后面对数据类别的统计，我们直接将数据和标签进行合并处理
    # 对于一个树的创建，因为我们事先是不知道这个树到底有多长，并且这个树到底要怎么去分的，因此肯定是需要使用递归的方法动态创建kd树
    def create_kd_tree(self, data_labels, min_dim, max_dim):
        # 如果没有数据了，那么就代表所有的数据都被分完了，这个时候就退出kd树的创建
        # 这里也是kd树递归结束的出口
        if len(data_labels) == 0:
            return  # 这里其实默认就返回了None
        # 这里是进行切分：对data_labels的min_dim维度进行排序，并且得到排序后的索引
        data_sorted = data_labels[np.argsort(data_labels[:, min_dim])]
        # 并且根据排序后的索引，得到排序后的数据
        # 得到排序后的数据的中间位置的索引
        data_middle_index = data_sorted.shape[0] // 2
        return Node(data_sorted[data_middle_index],
                    self.create_kd_tree(data_sorted[:data_middle_index], (min_dim + 1) % max_dim, max_dim),
                    self.create_kd_tree(data_sorted[data_middle_index + 1:], (min_dim + 1) % max_dim, max_dim),
                    min_dim)

    # 使用kd数来进行预测
    def use_kd_to_predict(self, data, k, p_distance):
        # 创建一个列表：用于后面创建堆对象
        top_k = [(-np.inf, None)] * k

        def find(node):
            # 如果该节点是空：表示已经遍历到树的末尾了，这个时候就返回
            if node is None:
                return
            # 根据当前输入的数据，来寻找最近的叶子结点
            # 根据每个维度上的数据进行判断，找到离该数据最近的叶子节点
            data_diff = data[node.dim] - node.data[node.dim]
            # 使用if 的条件表达式：如果比中间值小，那么就判断为左节点，否则判断为右节点
            find(node.left_node if data_diff < 0 else node.right_node)
            # 一直找到最近的节点后，根据该节点计算他们之间的距离
            # 因为kd树中的所有数据都是和label进行整合后的，因此舍去最后一个数据
            distance = p_distance(data.reshape((1, -1)), node.data.reshape((1, -1))[:, :-1])[0]
            # 将计算出来的距离存放到堆中，并且存入相应的标签
            # 因为这是一个最小堆，因此父类节点的数据肯定是小于等于子类节点的数据，因此需要对距离取负号
            # 这样可以使得最小的距离变成最大的数据5
            # 如果我想使用一个指定大小并且可以自动弹出我不想要的数据的容器 -> 那么我就可以选择堆
            heapq.heappushpop(top_k, (-distance, node.data[-1]))

            # 如果一旦出现存储的数据中，有比里面最大的数据还要小的数据，就去遍历该节点的另外一个子节点，寻找最近的距离
            # 直到我的堆中所有的数据都比节点的距离小
            if -top_k[0][0] > abs(data_diff):
                find(node.right_node if data_diff < 0 else node.left_node)

            # 得到栈中前k个最大数据
            # 得到所有存放在堆中数据的标签

        find(self.root)
        top = [value[1] for value in heapq.nlargest(k, top_k)]
        return top


class KNN(object):
    def __init__(self, data, labels, k, p):
        # 传入数据，以及想要k值(k在线性扫描中表示最近的k的个数，在kd树中表示堆中存放的距离值的个数)
        # p 表示距离度量中的 p ：一般去p为2(表示使用欧式距离)
        self.data = data
        self.labels = labels
        self.k = k
        self.p = p
        self.kd_tree = None  # 表示开始的时候没有构建kd树

    def p_distance(self, x, y):
        return np.sum(abs(x - y) ** self.p, -1) ** (1 / self.p)

        # 使用kd树的方式来进行搜索

    def kd_tree_val(self, tests_x):
        if self.kd_tree is None:
            self.kd_tree = KDTree(np.concatenate([self.data, self.labels.reshape((-1, 1))], -1))
        result_kind = []
        for test_x in tqdm(tests_x):
            top_k = self.kd_tree.use_kd_to_predict(test_x, self.k, self.p_distance)
            result_kind.append(self.vote(top_k))
        return result_kind

    # 调用判决函数：
    def vote(self, data):
        # 创建一个列表
        count = {}
        max_value, predictions = 0, 0
        for kind in data:
            # 这里表示提取字典中 kind 键对应的 value 值
            # 如果没有kind，则默认输出0，否则输出kind键对应的value,并且将该 value 重新赋给 kind 键的值
            count[kind] = count.get(kind, 0) + 1
            if count[kind] > max_value:
                max_value = count[kind]
                predictions = kind
        return predictions


if __name__ == '__main__':
    digits = datasets.load_digits()
    data_ = digits.data
    labels_ = digits.target
    x_train, x_test, y_train, y_test = train_test_split(data_, labels_, test_size=0.2, random_state=42)
    knn = KNN(x_train, y_train, 5, 2)
    y_predictions = knn.kd_tree_val(x_test)
    print('预测准确率为：{}%.'.format((y_predictions == y_test).mean() * 100))
