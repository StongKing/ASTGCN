import os
import numpy as np
import argparse
import configparser

# 从数据中采集需要的数据的索引
# 传过来的是sequence_length = date_sequece.shape[0], num_of_depend = num_of_hours
def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    # num-of-depend * unit 就是需要使用的历史数据小时数，再乘以 points_per_hour, 就是使用多少条历史数据，来预测未来的数据
    """
    Parameters
    ----------
    sequence_length: int, length of all history data 历史序列长度
    num_of_depend: int, 看依赖于周，日还是小时
    label_start_idx: int, the first index of predicting target 预测序列的第一个位置
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1 单元长度，按小时来计算，周就是168，日就是24，小时就是1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    """
    # 如果每小时采样点计算出错，或者发生别的错误导致小于0则报错
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    # 如果开始预测的位置加上预测序列的长度大于数据总长度那么返回空
    if label_start_idx + num_for_predict > sequence_length:
        return None
    # 选择训练数据，并设置训练数据索引列表
    x_idx = []
    # 选择多长时间（1周/1天/1个小时）作为训练数据
    # 进行一次采样, 将数值1,2,，...,num_of_depend 赋给 i
    for i in range(1, num_of_depend + 1):
        # 开始位置：从 points_per_hour * units * i 位置开始，如果units = num_of_depend
        start_idx = label_start_idx - points_per_hour * units * i
        # 结束位置：选择前 num_for_predict 个位置
        end_idx = start_idx + num_for_predict
        # end_idx = label_start_idx + num_for_predict
        # 这里的 end_ix是不是应该=label_start_idx + num_for_predict，因为label_start_idx是预测开始的点
        # 在后面有修改，所以这里不用提问
        # 知道起始位置大于等于0，再开始采样
        if start_idx >= 0:
            # 添加这一次采样开始和采样结束的位置（索引）存入列表
            x_idx.append((start_idx, end_idx))
        else:
            return None
    # 如果列表的长度不等于 num_of_depend ,即采样失败
    if len(x_idx) != num_of_depend:
        return None
    # 最后将列表元素倒序输出，因为对于后面append的数据，我们读取index的时候要先从最新的开始读取
    return x_idx[::-1]

# 组织样本集合sample
def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    
    # label_start_idx 开始的索引位置，用这之前的数据来作x，分界线后的数据作y，要大于初始数据采样点
    # label_start_idx + num_for_predict 不能大于数据的总长度
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    # 初始化数据样本
    week_sample, day_sample, hour_sample = None, None, None

    # 如果 分割线+预测数据长度大于 数据总长度，那么返回空，最后一个None表示的是 sample_y也返回空，即预测部分数据
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # 如果 选择以num_of_weeks，即周的时间长度来进行预测
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    # 如果 选择以num_of_hours，即小时的时间长度来进行预测
    if num_of_hours > 0:
        # 小时指标，根据采样search_data函数得到采样开始位置索引和结束位置索引
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        # 将其中一次的训练数据存入hour_sample，np.concatenate用于连接多个数组
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)
    # 获取对应其中一次的预测数据
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    # 返回一次的训练数据集和对应的预测数据集
    return week_sample, day_sample, hour_sample, target

#读取源数据并生成数据集
def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, save=False):
    # graph_signal_matrix_filename = ./data/PEMS08/PEMS08.npz
    # points_per_hour 表示的是每5分钟采样一次，一小时采样12次(要换成自己的数据就是每10分钟采样一次)
    # num_for_predict 预测的时间长短
    # num_for_weeks 是否采用一周的时间来预测
    # num_for_day 是否采用一天的时间来预测
    # num_for_hours 是否采用一小时的时间来预测
    # 对应的参数配置在PEMS08_astgcn.conf文件里
    """
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    """
    # 加载源文件
    # 加载一个名为 PEMS08.npz 的 NumPy 压缩文件（数组文件），并从中提取键为 'data' 的数组。得到的data_seq是一个
    data_seq = np.load(graph_signal_matrix_filename)['data']
    # (sequence_length, num_of_vertices, num_of_features)
    # print("Shape of the data_seq:", data_seq.shape)
    # print("First few entries in data_seq:", data_seq[:5])
    # Shape of the data_seq: (17856, 170, 3)

    all_samples = []
    # 循环data_seq.shape[0]=17856条数据
    for idx in range(data_seq.shape[0]):
        # 断点1
        # 组织样本集合sample，这里传入的idx即为label_start_idx，从此处得到每一次的训练数据和预测数据
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour) # 这里得到的sample有4个组，分别表示week_sample,day_sample,hour_sample,以及target,如果没有获取到样本则返回None
        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue
        # 将采样的数据赋给对应的week_sample/day_sample/hour_sample,将sample[3]赋给target作为预测数据
        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        # hour_sample shape: (1, N, F, T) = (1, 170, 3, 12), N表示传感器数，F表示特征数，T表示时间段
        # before: hour_sample shape = (12, 170, 3)
        # 这里的操作是扩充数据维度，因为每次提取的只是一个样本的数据
        if num_of_hours > 0:
            # 先进行了维度扩展变成(1,12,170,3),再进行了一次维度调换(1,170,3,12)
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            # 将扩展后的训练样本存入sample
            sample.append(hour_sample)
        # 对预测样本进行同样的操作, 但是只取三个feature里的第一个feature:flow,因为需要预测的是这个特征
        # target shape: (1, 170, 12)
        # target shape before: (12,170,3)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)
        # 保存了当前在data数据中的索引，并扩展为2维向量，保存到sample中，但是这里为什么要扩充
        # time_sample shape: (1, 1)
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(sample)
        # sample：[(week_sample), (day_sample), (hour_sample)训练样本, target预测样本, time_sample，分割线位置]
        # = [(1,N,F,Tw), (1,N,F,Td), (1,N,F,Th), (1,N,Tpre), (1,1)]
    # 到这里 all_sample个数其实比样本少了23个,是因为开始完整的一个训练样本+预测样本正好24个,按照窗口滑动所以少了24-1个

    # 对数据集进行切割，60% 作为训练，20% 验证，20%进行测试
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    # {list: 3}
    # [(B,N,F,Tw), (B,N,F,Td), (B,N,F,Th), (B,N,Tpre), (B,1)]
    # [(10699, 170, 3, 12), (10699, 170, 12), (10699, 1)]
    # 训练数据集，会生成三个数组一个是 x 一个是 y 还有一个index
    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]
    # 验证数据集
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    # 测试数据集
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    # 取出x
    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)
    # 取出y
    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]
    # 取出index
    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]
    # 为了加快训练速度和训练准确率需要对x进行 normalization，stats返回的是三个特征的均值和方差
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)
    # 存储所有数据到all_data字典里
    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])
    # 将文件存到一个压缩文件
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
            num_of_weeks)) + '_astcgn'
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data

#归一化数据
def normalization(train, val, test):
    """
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    """
    # 比较train和val数组从第二个维度到最后一个维度的大小是否相同，然后比较val和test数组的相应维度。使用and运算符来确保两个比较都为真。
    #
    # 这行代码的目的是确保train、val和test三个数据集中，除了第一个维度（通常是批次大小或样本数）之外，其他所有维度的大小都是相同的。这在机器学习和数据处理中非常重要，因为模型通常需要固定大小的输入。
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    # 对三个特征求均值，计算了train数组中所有样本、所有传感器在所有时间段上的平均特征值
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    # 计算方差
    std = train.std(axis=(0, 1, 3), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)
    # 对特征去均值归一化
    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)
    # 返回三种数据集的均值和方差，并返回归一化之后的数据集
    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


# prepare dataset 读取参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./configurations/PEMS08_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()

our_config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
our_config.read(args.config)

data_config = our_config['Data']
training_config = our_config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if our_config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
# num_of_vertices = int(data_config['num_of_vertices'])
# points_per_hour = int(data_config['points_per_hour'])
# num_for_predict = int(data_config['num_for_predict'])
# 数据文件名
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
data = np.load(graph_signal_matrix_filename)

# 查看文件中包含的所有数组的名称
print("数组名称:", data.files)
    
# 假设我们知道数组的名称，例如 'data'
graph_signal_matrix = data['data']  
    
# 打印数组的内容
print("图信号矩阵内容:\n", graph_signal_matrix)
    
# 查看数组的形状、数据类型等属性
print("形状:", graph_signal_matrix.shape)
print("数据类型:", graph_signal_matrix.dtype)

all_data = read_and_generate_dataset(graph_signal_matrix_filename, 0, 0, num_of_hours, num_for_predict,
                                     points_per_hour=points_per_hour, save=True)
