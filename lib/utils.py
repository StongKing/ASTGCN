import os
import numpy as np
import torch
import torch.utils.data
from scipy.sparse.linalg import eigs
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np, masked_mae_test, masked_rmse_test


# 这个函数用于将标准化的数据重新缩放回原始范围。它接受标准化后的数据 x，以及用于标准化的 mean 和 std 参数，然后计算 x = x * std + mean，最后返回重新缩放后的数据。
def re_normalization(x, mean, std):
    x = x * std + mean
    return x

# 这个函数执行最小-最大归一化，将数据缩放到 [-1, 1] 区间。它接受原始数据 x 和数据的最大值 _max 和最小值 _min，然后计算 x = (x - _min) / (_max - _min)，接着将其缩放到 [-1, 1] 区间。
def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x

# 这个函数是 max_min_normalization 的逆过程，它将归一化到 [-1, 1] 区间的数据恢复到原始范围。
def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

# 这个函数用于从CSV文件中加载图的邻接矩阵。如果提供的文件名包含 .npy，则直接加载 .npy 文件；
# 否则，从CSV文件中读取边信息并构建邻接矩阵 A 和距离矩阵 distaneA。如果提供了 id_filename，则使用该文件中的ID映射来构建邻接矩阵。
def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    # 如果之前生成了.npy文件
    if 'npy' in distance_df_filename:
        # 使用 np.load 函数加载 .npy 文件中的邻接矩阵 adj_mx
        adj_mx = np.load(distance_df_filename)
        # 返回加载的邻接矩阵 adj_mx 和 None，因为不需要构建距离矩阵。
        return adj_mx, None
    else:
        import csv
        # 创建一个形状为 (num_of_vertices, num_of_vertices) 的零矩阵 A，用于存储邻接矩阵，数据类型为 float32。
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        # 创建一个形状为 (num_of_vertices, num_of_vertices) 的零矩阵 distaneA，用于存储距离矩阵，数据类型为 float32。
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        # 打印邻接矩阵和距离矩阵的形状，用于调试。
        print(A.shape)
        print(distaneA.shape)
        # 检查是否提供了 id_filename 参数，如果提供了，表示需要使用ID映射。
        if id_filename:
            # 打开 id_filename 文件进行读取。读取文件内容，创建一个字典 id_dict，将文件中的ID映射到从0开始的索引。
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            # 打开包含边信息的CSV文件进行读取。
            with open(distance_df_filename, 'r') as f:
                # 读取并丢弃CSV文件的第一行（通常是标题行）。
                f.readline()
                # 创建一个 csv.reader 对象，用于读取CSV文件。
                reader = csv.reader(f)
                # 遍历CSV文件中的每一行。
                for row in reader:
                    # 检查每行是否包含3个元素（两个节点ID和一个距离），如果不是，则跳过这一行。
                    if len(row) != 3:
                        continue
                    # 将CSV文件中的每行数据转换为整数 i 和 j（节点ID），以及浮点数 distance（节点间的距离）
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    # 在邻接矩阵 A 中，将对应节点的连接设置为1，表示这两个节点之间有边连接
                    A[id_dict[i], id_dict[j]] = 1
                    # 在距离矩阵 distaneA 中，设置对应节点间的距离
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:
            # 如果没有提供 id_filename 参数，表示节点ID直接从0开始
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    # print(i)
                    # print(j)
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


# 这个函数计算图的缩放拉普拉斯矩阵。它接受邻接矩阵 W 作为输入，计算拉普拉斯矩阵 L = D - W，其中 D 是度矩阵。
# 然后，它通过计算最大特征值 lambda_max 并使用公式 (2 * L) / lambda_max - I 来缩放拉普拉斯矩阵，其中 I 是单位矩阵。
def scaled_Laplacian(W):
    """
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    """

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    # 拉普拉斯矩阵定义：L = D - W(邻接矩阵)
    L = D - W

    # shift变换将特征值对角矩阵值转换到[-1, 1]
    # 为什么值区间转换成[-1, 1]?
    # Chebyshev多项式作为GCN卷积核有什么好处？
    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

# 这个函数计算切比雪夫多项式，直到 K 阶。它接受缩放后的拉普拉斯矩阵 L_tilde 和阶数 K，然后计算从 T_0 到 T_{K-1} 的切比雪夫多项式列表。
def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    """

    N = L_tilde.shape[0]

    # T_0: I, T_1: scaled_Laplacian matrix
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    # T_k = 2 * L_tilde * T_{k-1} - T_{k-2}
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

# 加载图数据 ，特别是针对PEMS数据集。它处理数据，将输入和目标数据归一化到 [-1, 1] 区间，并构建数据加载器 DataLoader。
def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size,
                            shuffle=True):
    """
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    """

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
                                num_of_weeks)) + '_astcgn'

    print('load file:', filename, '.npz')

    file_data = np.load(filename + '.npz')

    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------加载数据，张量化
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std

# 这个函数计算模型在验证集上的损失。它接受模型、验证数据加载器、损失函数、掩码标志、缺失值、摘要写入器和当前周期作为输入，计算并返回验证损失。
def compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, limit=None):
    """
    for rnn, compute mean loss on validation set
    :param net: model 要评估的模型（通常是神经网络）
    :param val_loader: torch.utils.data.utils.DataLoader 验证集的数据加载器，用于迭代验证数据。
    :param criterion: torch.nn.MSELoss 损失函数，用于计算模型输出和目标之间的差异。
    :param sw: tensorboardX.SummaryWriter TensorBoard的SummaryWriter对象，用于记录训练和验证过程中的指标。
    :param masked_flag: 一个标志，指示是否需要使用掩码损失函数。
    :param missing_value：在掩码损失函数中使用的缺失值。
    :param global_step: int, current global_step
    :param limit: int, limit：可选参数，用于限制验证过程中处理的批次数量。
    :return: val_loss
    epoch：当前的迭代周期（epoch）。
    """




    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss

# 这个函数评估模型在测试集上的性能，计算MAE、RMSE和MAPE指标
def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    """
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.

    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    """

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        test_loader_length = len(test_loader)

        test_target_tensor = test_target_tensor.cpu().numpy()

        prediction = []  # 存储所有batch的output

        for batch_index, batch_data in enumerate(test_loader):

            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert test_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            print()
            if sw:
                sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
                sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
                sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)

# 这个函数用于预测并保存结果。它接受模型、数据加载器、目标张量、全局步骤、度量方法、均值和标准差、参数路径和类型作为输入，预测数据，计算误差，并将结果保存到文件中。
def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std,
                                    params_path, type):
    """
    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param global_step: int, epoch
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    """
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                rmse = masked_rmse_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            else:
                mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.4f' % (mae))
            print('RMSE: %.4f' % (rmse))
            print('MAPE: %.4f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        if metric_method == 'mask':
            mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        else:
            data_target_tensor_sliced = data_target_tensor[:, :, :]
            prediction_sliced = prediction[:, :, :]
            mae = mean_absolute_error(data_target_tensor_sliced.reshape(-1, 1), prediction_sliced.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor_sliced.reshape(-1, 1), prediction_sliced.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(data_target_tensor_sliced.reshape(-1, 1), prediction_sliced.reshape(-1, 1), 0)
        print('all MAE: %.4f' % (mae))
        print('all RMSE: %.4f' % (rmse))
        print('all MAPE: %.4f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)


