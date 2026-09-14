import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGCN_r import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, \
    predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae, masked_mse, masked_rmse


# Python的argparse模块，它是一个用于构建命令行接口的库。argparse允许定义命令行参数，这些参数可以被用户在执行脚本时指定。
parser = argparse.ArgumentParser()
# 这是向ArgumentParser对象添加一个命令行参数的方法,这里添加了一个名为--config的参数
parser.add_argument("--config", default='./configurations/DIVVY_without_0_astgcn.conf', type=str,
                    help='configuration file path')
# 这行代码调用ArgumentParser对象的parse_args方法来解析命令行输入。解析后的结果被存储在args变量中，它是一个Namespace对象，包含了所有的命令行参数。
args = parser.parse_args()

# 读取参数配置，使用了Python的configparser模块来读取和解析配置文件

# 这行代码创建了ConfigParser类的一个新的实例。ConfigParser用于读取配置文件，它会自动处理文件中的键值对，并将它们组织成字典形式，方便访问。
config = configparser.ConfigParser()
# 告诉用户程序正在读取的配置文件路径。args.config是从上一段代码中的命令行参数解析得到的配置文件路径。
print('Read configuration file: %s' % args.config)
# ConfigParser对象去读取由args.config指定的配置文件。read方法读取文件并解析其内容，将每个部分（section）和其下的键值对存储在内部结构中。
config.read(args.config)

# 从 config 对象中获取名为 [Data] 的部分的所有配置项。如果成功获取，data_config 将是一个字典，包含了 [Data] 部分下所有的键值对
data_config = config['Data']
# 获取 [Training] 部分的所有配置项，并将它们存储在 training_config 字典中
training_config = config['Training']

# 从data_config字典中读取键adj_filename对应的值。假设data_config是通过正确方式从config['Data']转换来的，这个键对应的值可能是一个文件路径，指向存储邻接矩阵的文件
adj_filename = data_config['adj_filename']
# 从data_config字典中读取键graph_signal_matrix_filename的值，这个值通常是一个文件路径，指向存储图信号矩阵的文件
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
# 使用config.has_option方法来检查[Data]部分是否存在键id_filename
if config.has_option('Data', 'id_filename'):
    # 如果[Data]部分中存在id_filename，那么这行代码将获取其值。这个值可能是一个文件路径，指向存储节点ID的文件
    id_filename = data_config['id_filename']
else:
    id_filename = None

# 从配置文件的 [Data] 部分读取数据
# 从 data_config 字典中读取键 num_of_vertices 对应的值，并将其转换为一个整数。这个值代表图中顶点（vertices）的数量。
num_of_vertices = int(data_config['num_of_vertices'])
# 读取键 points_per_hour 对应的值，并将其转换为整数。这个值表示每个小时收集的数据点数量。
points_per_hour = int(data_config['points_per_hour'])
# 获取键 num_for_predict 的值，并将其转换为整数。这个值指的是预测任务中使用的预测时间的步长数。
num_for_predict = int(data_config['num_for_predict'])
# 读取键 len_input 对应的值，并将其转换为整数。这个值代表输入特征的时间序列长度，即模型在进行预测时考虑的历史时间步长数。
len_input = int(data_config['len_input']) # len_input
# 读取键 dataset_name 对应的值，这个值是一个字符串，表示数据集的名称
dataset_name = data_config['dataset_name']

# 读取模型训练相关的参数，并设置环境变量和设备配置
# 从 training_config 字典中读取键 model_name 对应的值，并将其存储在变量 model_name 中。这个值通常是一个字符串，表示要训练的模型的名称。
model_name = training_config['model_name']

# 读取键 ctx 对应的值，并将其存储在变量 ctx 中。ctx 通常代表计算上下文，可能是用于指定使用哪个GPU进行训练
ctx = training_config['ctx']
# 设置环境变量 CUDA_VISIBLE_DEVICES 的值为 ctx。这个环境变量用于控制哪些GPU对Python程序可见。通过设置这个环境变量，可以指定程序应该使用特定的GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
# 使用PyTorch的 torch.cuda.is_available() 函数来检查CUDA是否可用。如果可用，USE_CUDA 将被设置为 True，否则为 False。
USE_CUDA = torch.cuda.is_available()
# 创建了一个PyTorch设备对象 DEVICE，指定使用编号为0的CUDA设备。如果CUDA不可用，这个设备对象将不会指向有效的硬件。
DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')
# 打印出CUDA是否可用以及设备信息
print("CUDA:", USE_CUDA, DEVICE)

# 读取学习率的值，并将其转换为浮点数，用于调整模型权重的更新幅度。
learning_rate = float(training_config['learning_rate'])
# 读取训练的总轮数（epoch），并将其转换为整数。一个epoch指的是整个数据集在训练过程中被遍历了一次。
epochs = int(training_config['epochs'])
# 读取开始训练的epoch，并将其转换为整数。这可以用于从先前的训练状态恢复。
start_epoch = int(training_config['start_epoch'])
# 读取每个批次的样本数量，并将其转换为整数。Batch size决定了每次训练迭代中用于更新模型权重的数据量。
batch_size = int(training_config['batch_size'])
# 读取模型训练或预测中使用的时间序列数据的周数，天数，小时数
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
# 将变量 time_strides 设置为 num_of_hours 的值，用于定义时间序列数据的时间步长
time_strides = num_of_hours
# 读取Chebyshev滤波器的数量，并将其转换为整数。Chebyshev滤波器用于图信号处理。
nb_chev_filter = int(training_config['nb_chev_filter'])
# 读取时间滤波器的数量，并将其转换为整数。时间滤波器用于处理时间序列数据。
nb_time_filter = int(training_config['nb_time_filter'])
# 读取输入通道数，并将其转换为整数。这通常用于定义模型输入层的通道数量。在本实例中为第一个特征。
in_channels = int(training_config['in_channels'])
# 读取模型中块（block）的数量，并将其转换为整数。与ASTGCN模型结构中的重复单元有关
nb_block = int(training_config['nb_block'])
# 读取参数 K 的值，并将其转换为整数。K 的具体含义取决于模型和任务，在本例子中为切比雪夫相关参数。
K = int(training_config['K'])
# 读取损失函数的名称。字符串，表示用于模型训练的损失函数类型
loss_function = training_config['loss_function']
# 读取评估模型性能的指标方法的名称。字符串，表示用于评估模型的指标类型
metric_method = training_config['metric_method']
# 读取缺失值的表示，并将其转换为浮点数。在处理数据时，这个值用于代替缺失的数据点
missing_value = float(training_config['missing_value'])

# 这段代码定义了一个文件夹路径 folder_dir，该路径用于存储与训练模型相关的参数、日志或模型权重等。然后，它使用 os.path.join 函数来构建一个完整的文件路径 params_path，该路径指向模型参数文件应该被保存的目录。
# 创建一个包含多个参数的文件夹名称。格式化操作通过 %s、%d 和 %e 占位符完成，分别代表字符串、整数和浮点数类型
folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (
    model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
# print('folder_dir:', folder_dir)
# 使用 os.path.join 函数来连接几个字符串，形成一个完整的文件路径。这里，它将 'experiments' 作为基目录，然后是之前从配置文件中读取的 dataset_name，最后是上面创建的 folder_dir 文件夹名称。
params_path = os.path.join('experiment', dataset_name, folder_dir)
# 打印出最终的 params_path，以便于用户知道模型参数文件将被保存在哪里
print('params_path:', params_path)

'''
调用负责加载图数据和将其组织成适合模型训练的数据加载器。函数的参数包括:
graph_signal_matrix_filename: 图信号矩阵文件的名称，它包含了图的节点特征。
num_of_hours, num_of_days, num_of_weeks: 这些参数定义了如何从图信号矩阵中分割数据，用于创建时间序列数据的滑动窗口。
DEVICE: 指定了模型和数据应该运行在的设备（CPU或GPU）。
batch_size: 训练时每个批次的样本数量。
并返回
train_loader: 训练数据的加载器。
train_target_tensor: 训练数据的目标张量。
val_loader: 验证数据的加载器。
val_target_tensor: 验证数据的目标张量。
test_loader: 测试数据的加载器。
test_target_tensor: 测试数据的目标张量。
_mean 和 _std: 数据特征的均值和标准差，用于归一化处理。
'''
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = \
    load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)
'''
函数调用负责从邻接文件中获取邻接矩阵。函数的参数包括：
adj_filename: 邻接文件的名称，它包含了图的邻接信息。
num_of_vertices: 图中顶点的数量。
id_filename: 可选参数，如果提供，可能包含了图顶点的额外信息。
函数返回两个对象：
adj_mx: 图的邻接矩阵。
distance_mx: 图的距离矩阵，可能用于某些类型的图卷积网络。
'''
adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

'''
make_model(...): 这个函数调用负责根据提供的参数创建模型。函数的参数包括：
DEVICE: 模型运行的设备。
nb_block: 模型中的块数量。
in_channels: 输入通道数。
K: 可能是某种采样或池化操作的参数。
nb_chev_filter: Chebyshev滤波器的数量。
nb_time_filter: 时间滤波器的数量。
time_strides: 时间步长。
adj_mx: 图的邻接矩阵。
num_for_predict: 预测所需的时间点数量。
len_input: 输入序列的长度。
num_of_vertices: 图中顶点的数量。
函数返回一个对象 net，它是根据提供的参数配置的模型。
'''
# 调用 make_model 的函数来创建一个基于切比雪夫核的图神经网络学习模型
net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices)

# 定义了一个名为 train_main 的函数，它负责在开始模型训练之前处理参数存储路径 params_path
def train_main():
    # 如果 start_epoch 等于0（通常表示这是训练的开始），并且 params_path 不存在，那么
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        # 使用 os.makedirs(params_path) 创建一个新的参数目录
        os.makedirs(params_path)
        # 打印消息，告知用户已创建参数目录
        print('create params directory %s' % params_path)
    # 如果 start_epoch 等于0，但 params_path 已存在，那么
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        # 使用 shutil.rmtree(params_path) 删除现有的参数目录
        shutil.rmtree(params_path)
        # 使用 os.makedirs(params_path) 创建一个新的参数目录
        os.makedirs(params_path)
        # 打印消息，告知用户已删除旧目录并创建了新的参数目录
        print('delete the old one and create params directory %s' % (params_path))
    # 如果 start_epoch 大于0（表示训练不是从头开始，可能是从之前的检查点恢复），并且 params_path 存在，那么
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        # 打印消息，告知用户将从现有的参数目录恢复训练
        print('train from params directory %s' % params_path)
    # 如果以上条件都不满足
    else:
        # 引发 SystemExit 异常，并打印错误消息 'Wrong type of model!'，这通常意味着模型类型或参数路径存在问题，训练无法继续
        raise SystemExit('Wrong type of model!')
    
    # 打印参数列表：打印出所有训练所需的参数，包括设备信息、模型参数、批处理大小、文件路径、训练轮数等
    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    # 初始化 masked_flag，这个标志可能用于指示是否使用掩码损失函数
    masked_flag = 0
    # 初始化一个损失函数 criterion，使用L1损失（即平均绝对误差），并将其发送到 DEVICE
    criterion = nn.L1Loss().to(DEVICE)
    # 初始化 criterion_masked 为 masked_mae，这可能是一个自定义的掩码平均绝对误差损失函数
    criterion_masked = masked_mae
    # 根据配置文件中指定的损失函数类型，设置 criterion_masked 并调整 masked_flag
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse  # nn.MSELoss().to(DEVICE)
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'mse':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    # 创建一个优化器 optimizer，使用Adam算法，并将其应用于模型 net 的参数，设置学习率为 learning_rate
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # 初始化 SummaryWriter，用于将训练过程中的信息记录到日志中，以便可以在 TensorBoard 中查看。
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    # 打印出模型 net 的结构
    print(net)
    # 打印模型的 state_dict（参数字典）的标题
    print('Net\'s state_dict:')
    # 并初始化 total_param 用于计算模型的总参数数量
    total_param = 0
    # 遍历模型的所有参数，打印每个参数的名称和大小，并计算总参数数量
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    # 打印出模型的总参数数量
    print('Net\'s total params:', total_param)
    # 打印优化器的状态字典
    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    # 初始化全局步骤计数器 global_step，最佳epoch best_epoch 和最佳验证损失 best_val_loss
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    # 记录训练开始的时间
    start_time = time()
    # 如果从非零epoch恢复训练，加载之前保存的模型参数，大于0，这意味着训练不是从头开始，而是从先前的某个epoch恢复
    if start_epoch > 0:
        # os.path.join 用于将 params_path（存储模型参数的目录路径）与由当前的 start_epoch 值决定的文件名连接起来。'epoch_%s.params' % start_epoch 是字符串格式化表达式，它创建了一个包含当前 start_epoch 值的文件名
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        # 加载之前保存的模型参数到模型 net 中。torch.load 函数用于从 params_filename 指定的文件中读取模型的状态字典（state_dict），然后 net.load_state_dict 方法将这些参数加载到模型 net 的参数中
        net.load_state_dict(torch.load(params_filename))
        # 打印一条消息，显示从哪个epoch开始恢复训练
        print('start epoch:', start_epoch)
        # 打印一条消息，显示模型权重加载自哪个文件
        print('load weight from: ', params_filename)

    # train model
    # 开始训练循环，从 start_epoch 到 epochs
    for epoch in range(start_epoch, epochs):
        # 对于当前的每个epoch，生成一个用于存储模型参数的文件路径
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        # 计算当前epoch的验证损失。如果设置了 masked_flag，则使用 criterion_masked 掩码损失函数；否则，使用普通的 criterion 损失函数。
        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)
        # 如果当前epoch的验证损失小于之前记录的 best_val_loss，则更新最佳验证损失和最佳epoch，并保存当前模型的参数到 params_filename 指定的文件中
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)
        # 将模型设置为训练模式，以确保dropout等随机层在训练时按预期工作
        net.train()  # ensure dropout layers are in train mode
        
        # 开始遍历训练数据加载器 train_loader 中的每个批次
        for batch_index, batch_data in enumerate(train_loader):
            # 从 train_loader 中获取当前批次的数据和标签
            encoder_inputs, labels = batch_data
            # 在每次反向传播前，清除（归零）当前的梯度信息
            optimizer.zero_grad()
            # 执行模型的前向传播，计算输出
            outputs = net(encoder_inputs)
            # 根据是否设置了 masked_flag，计算当前批次的损失
            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)
                # loss = criterion(outputs, labels) +  ((start_epoch+1)/epochs)*0.05*torch.mean(torch.abs(torch.sum(labels, dim=1) - torch.sum(outputs, dim=1)))/68
                # loss = criterion(outputs, labels) + ((start_epoch+1)/epochs)*0.05 * torch.mean(torch.abs(torch.sum(labels, dim=1) - torch.sum(outputs, dim=1)))
            # 执行反向传播算法，计算损失相对于模型参数的梯度
            loss.backward()
            # 根据计算出的梯度，更新模型的权重
            optimizer.step()
            # 获取当前批次的损失值
            training_loss = loss.item()
            # 增加全局步骤计数器
            global_step += 1
            # 使用 SummaryWriter 记录当前批次的训练损失到TensorBoard日志
            sw.add_scalar('training_loss', training_loss, global_step)
            # 如果全局步骤是1000的倍数，打印当前的训练进度和时间
            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (
                    global_step, training_loss, time() - start_time))
    # 在训练循环结束后，打印出整个训练过程中最佳的epoch
    print('best epoch:', best_epoch)
   
    # apply the best model on the test set
    # 调用 predict_main 函数，使用最佳模型在测试集上进行预测，并根据 metric_method 计算并输出性能指标。_mean 和 _std 用于数据的标准化处理，而 'test' 指定了数据集类型为测试集
    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, 'test')


'''这个函数通常在模型训练完成后调用，用于在测试集或其他数据集上评估模型的性能，并将预测结果持久化存储，以便于后续分析或报告
'''
# 定义了一个名为 predict_main 的函数，其目的是使用训练好的模型在给定的数据集上进行预测，并将结果保存起来
def predict_main(global_step, data_loader, data_target_tensor, metric_method, _mean, _std, type):
    """
    :param global_step: int：一个整数，通常表示模型是从哪个全局步骤或epoch的检查点恢复的
    :param data_loader: torch.utils.data.utils.DataLoader 一个 DataLoader 对象，用于批量加载数据集
    :param data_target_tensor: tensor 一个张量，包含了数据集的目标值或标签
    :param metric_method: tensor 一个张量或字符串，表示用于评估模型性能的度量方法
    :param _mean: (1, 1, 3, 1) 数据的均值和标准差，它们可能用于数据的归一化或反归一化
    :param _std: (1, 1, 3, 1) 数据的标准差
    :param type: string 一个字符串，指定了数据集的类型，如 'train'、'val' 或 'test'
    :return:
    """
    # 基于全局步骤 global_step 创建模型参数文件的路径
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    # 打印一条消息，指示模型将从哪个文件加载权重
    print('load weight from:', params_filename)
    # 从指定的文件路径加载模型的参数。这里假设 net 是一个PyTorch模型实例，且 params_filename 指向一个包含了模型状态字典的文件
    net.load_state_dict(torch.load(params_filename))
    # 调用 predict_and_save_results_mstgcn 函数来进行预测，并将结果保存起来
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std,
                                    params_path, type)
    '''
    使用 net（加载了参数的模型）和 data_loader 进行预测。
    使用 metric_method 来评估模型的性能。
    使用 _mean 和 _std 对数据进行反归一化，以便得到原始尺度的预测结果。
    将预测结果和可能的性能评估指标保存到文件中。保存的位置由 params_path 指定，而 type 参数可能用于指定保存结果的文件名或目录。
    '''


'''
在Python中，每当你导入一个模块时，Python解释器会自动设定一个特殊的变量 __name__。
当你直接运行一个Python脚本时，Python会设置 __name__ 为 "__main__"。
因此，这个条件检查是用来确定代码是否在作为独立脚本运行，而不是被导入到另一个模块中
这种模式的好处是，即使脚本被导入到其他模块中，train_main() 函数也不会被执行，
这可以避免在导入时不小心执行训练代码。只有当你直接运行这个脚本时，train_main() 函数才会执行
使得这个脚本既可以被当作程序执行，也可以被导入为模块使用，而不会导致立即执行训练代码
'''
# 用于检查当前脚本是否作为主程序运行。如果这个条件为真，那么它将调用 train_main() 函数
if __name__ == "__main__":
    train_main()

    # predict_main(77, test_loader, test_target_tensor,metric_method, _mean, _std, 'test')
