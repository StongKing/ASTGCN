'''
这段代码定义了几个深度学习模型类和创建模型实例的函数，
主要用于构建一个图卷积网络（ASTGCN），它结合了空间和时间注意力机制。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial

'''
class Spatial_Attention_layer(nn.Module)空间注意力层:计算空间注意力分数，用于加权图中的节点
class cheb_conv_withSAt(nn.Module) 带空间注意力的Chebyshev卷积，使用Chebyshev多项式作为卷积核，并将空间注意力应用于图卷积
class Temporal_Attention_layer(nn.Module) 时间注意力层：它计算时间注意力分数，用于加权时间序列数据
class cheb_conv(nn.Module) 标准的Chebyshev图卷积层，不包含注意力机制
class ASTGCN_block(nn.Module)定义了一个ASTGCN网络的基本块，它包含时间注意力层、空间注意力层、带空间注意力的Chebyshev卷积层和时间卷积层
class ASTGCN_submodule(nn.Module) 这个类构建了ASTGCN网络的主体，它包含多个ASTGCN块，并在最后使用一个卷积层来生成预测。
def make_model(...)这个函数根据提供的参数创建并初始化ASTGCN模型。它首先生成Chebyshev多项式，然后实例化ASTGCN子模块，并初始化模型参数
'''

'''
这个类实现了一个空间注意力机制，它通过学习输入数据的空间依赖性来计算注意力分数，
这些分数可以用于加权图中的节点，从而提高模型对空间特征的敏感度。
'''
# 定义了一个类 Spatial_Attention_layer，它继承自 nn.Module，是PyTorch中用于构建神经网络模块的基类
# 这个类用于计算空间注意力分数
class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """
    # 构造函数初始化模块，并调用基类的构造函数。 
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        # 定义参数矩阵 W1，其大小为 (num_of_timesteps,)，并且将其发送到 DEVICE
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        # 定义第二个参数矩阵 W2，其大小为 (in_channels, num_of_timesteps)，同样发送到 DEVICE
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        # 定义第三个参数向量 W3，其大小为 (in_channels,)，也发送到 DEVICE
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        # 定义一个参数张量 bs，其大小为 (1, num_of_vertices, num_of_vertices)，用作空间注意力的偏置项，并发送到 DEVICE
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        # 定义一个参数矩阵 Vs，其大小为 (num_of_vertices, num_of_vertices)，用作空间注意力的可学习矩阵，并发送到 DEVICE
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))
    
    '''
    定义前向传播函数，它接收输入 x，其形状为 (batch_size, num_of_vertices, in_channels, num_of_timesteps)，
    并返回一个形状为 (batch_size, num_of_vertices, num_of_vertices) 的注意力矩阵
    '''
    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        """
        # 计算左侧的注意力分数，通过连续的矩阵乘法操作
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        # 计算右侧的注意力分数，并通过转置操作调整形状
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        # 计算左右两侧分数的点积
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        # 通过激活函数和可学习矩阵 Vs 计算最终的空间注意力矩阵
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        # 对注意力矩阵进行归一化，使得每一行的和为1
        S_normalized = F.softmax(S, dim=1)
        # 返回归一化后的空间注意力矩阵
        return S_normalized

'''
这个类实现了一个结合了空间注意力机制的 K 阶 Chebyshev 图卷积层。
通过使用 Chebyshev 多项式作为图卷积核，它可以有效地捕捉图结构数据的局部特征。
空间注意力机制允许模型学习图中不同节点之间的重要性，从而提高模型的性能
'''
# 定义了一个名为 cheb_conv_withSAt 的类，它是 nn.Module 的子类，用于实现 K 阶 Chebyshev 图卷积，并且结合了空间注意力机制
class cheb_conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        # 构造函数初始化 Chebyshev 卷积层
        super(cheb_conv_withSAt, self).__init__()
        # 接收多项式阶数 K
        self.K = K
        # Chebyshev 多项式列表 cheb_polynomials 
        self.cheb_polynomials = cheb_polynomials
        # 输入通道数 in_channels 
        self.in_channels = in_channels
        # 输出通道数 out_channels
        self.out_channels = out_channels
        # DEVICE 是用来确定模型参数应该放置在CPU还是GPU上
        self.DEVICE = cheb_polynomials[0].device
        # 初始化一个 ParameterList，包含 K 个参数矩阵 Theta，每个矩阵的大小为 (in_channels, out_channels)，并且发送到 DEVICE
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
    '''
    定义前向传播函数，它接收输入 x，
    其形状为 (batch_size, num_of_vertices, in_channels, num_of_timesteps)，
    以及空间注意力矩阵 spatial_attention，
    并返回形状为 (batch_size, num_of_vertices, out_channels, num_of_timesteps) 的输出。
    '''
    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """
        # 获取输入 x 的形状信息
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        # 初始化一个空列表，用于存储每个时间步的输出
        outputs = []
        # 对于输入数据的每个时间步，提取图信号
        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            # 初始化输出张量，大小为 (batch_size, num_of_vertices, out_channels)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)
            # 
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N) # 第二个矩阵是L_titlde
                # 对于每个 Chebyshev 多项式，乘以空间注意力矩阵，得到加权的多项式矩阵
                T_k_with_at = T_k.mul(spatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)
                # 对每个多项式进行矩阵乘法操作，并将结果累加到输出张量
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
            # 将每个时间步的输出张量增加一个维度，以便可以沿着时间步进行连接
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        # 使用 torch.cat 连接所有时间步的输出，并应用 ReLU 激活函数，得到最终的输出
        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


'''
Temporal_Attention_layer 类的构造函数初始化时间注意力层。
它创建了几个参数矩阵和向量，这些参数将用于计算时间注意力分数
这个类实现了一个时间注意力机制，它通过学习输入数据的时间依赖性来计算注意力分数，
这些分数可以用于加权时间序列数据中的不同时间点，从而提高模型对时间特征的敏感度
'''
class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        # 一个 (num_of_vertices,) 形状的参数向量
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        # 一个 (in_channels, num_of_vertices) 形状的参数矩阵
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        # 一个 (in_channels,) 形状的参数向量
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        # 一个 (1, num_of_timesteps, num_of_timesteps) 形状的参数张量，用作时间注意力的偏置项
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        # 一个 (num_of_timesteps, num_of_timesteps) 形状的参数矩阵
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        # 前向传播函数首先获取输入 x 的形状信息
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # 计算左侧的时间注意力分数。首先，x 被转置为 (batch_size, num_of_timesteps, in_channels, num_of_vertices) 形状，然后通过两次矩阵乘法计算注意力分数。
        # print(x.shape)
        # print(self.U1.shape)
        # print(self.U2.shape)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        
        # 计算右侧的时间注意力分数，self.U3 与 x 进行矩阵乘法
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        # 计算左右两侧分数的点积，得到初步的时间注意力矩阵
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        # 通过激活函数和可学习矩阵 Ve 计算最终的时间注意力矩阵
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        # 对时间注意力矩阵进行归一化，使得每一列的和为1
        E_normalized = F.softmax(E, dim=1)
        # 返回归一化后的时间注意力矩阵
        return E_normalized


'''
cheb_conv 类的构造函数与 cheb_conv_withSAt 类似，
但只初始化了 Chebyshev 多项式和参数矩阵 Theta，没有初始化空间注意力相关的参数
它直接使用 Chebyshev 多项式和参数矩阵 Theta 来计算图卷积的输出
'''
class cheb_conv(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


'''
ASTGCN_block 类的构造函数初始化了一个 ASTGCN 块，
它包含了时间注意力层、空间注意力层、带空间注意力的 Chebyshev 图卷积层、时间卷积层以及残差连接层：
'''
class ASTGCN_block(nn.Module):

    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        '''
        self.TAt: 初始化时间注意力层。
        self.SAt: 初始化空间注意力层。
        self.cheb_conv_SAt: 初始化带空间注意力的 Chebyshev 图卷积层。
        self.time_conv: 初始化时间卷积层，使用 nn.Conv2d，设置输入通道数为 nb_chev_filter，
                        输出通道数为 nb_time_filter，时间步长为 time_strides。
        self.residual_conv: 初始化残差连接的时间卷积层，使用 nn.Conv2d，设置输入通道数为 in_channels，
                            输出通道数为 nb_time_filter，时间步长为 time_strides。
        self.ln: 初始化层归一化层 LayerNorm，作用在 nb_time_filter 通道上。
        '''
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # 将channel放到最后一个维度上

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        """
        # 前向传播函数首先获取输入 x 的形状信息
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # 然后计算时间注意力矩阵temporal_At
        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        # x_TAt: 将时间注意力应用于输入 x
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,
                                                                                               num_of_vertices,
                                                                                               num_of_features,
                                                                                               num_of_timesteps)
        # 计算空间注意力矩阵
        # SAt
        spatial_At = self.SAt(x_TAt)
        # spatial_gcn: 通过带空间注意力的 Chebyshev 图卷积层处理 x_TAt
        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)
        # 应用时间卷积层到空间图卷积的输出
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        
        # 创建残差连接，并应用层归一化和 ReLU 激活函数
        # x_residual: 计算残差连接的时间卷积输出。
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        # 通过 F.relu 应用 ReLU 激活函数，通过 self.ln 应用层归一化。
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        # 返回处理后的数据，它是时间卷积输出和残差连接输出的和
        return x_residual






# ASTGCN_submodule 类的构造函数初始化一个异构图卷积网络子模块，它包含多个 ASTGCN_block 块：
class ASTGCN_submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_for_predict, len_input, num_of_vertices):
        """
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides: num_of_hours=1
        :param cheb_polynomials:
        :param nb_predict_step:
        """

        super(ASTGCN_submodule, self).__init__()
    
        # self.BlockList: 使用 nn.ModuleList 初始化一个模块列表，首先添加一个 ASTGCN_block
        self.BlockList = nn.ModuleList([ASTGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                                     time_strides, cheb_polynomials, num_of_vertices, len_input)])
        # 向模块列表中添加更多的 ASTGCN_block 块，数量由 nb_block - 1 决定。这些块的输入通道数为 nb_time_filter，时间步长为1，因为它们是网络中的后续块。
        self.BlockList.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                                            cheb_polynomials, num_of_vertices, len_input // time_strides) for _ in range(nb_block - 1)])
        # 初始化最终的卷积层 self.final_conv，用于从最后一个图卷积块的输出中预测目标值。它使用 nn.Conv2d，设置输入通道数为 len_input / time_strides，输出通道数为 num_for_predict
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        # 设置模型的设备为 DEVICE，并将模型的所有参数发送到相应的设备（CPU或GPU）
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        """
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        """
        #
        # print(x.shape)

        # 前向传播函数首先遍历 BlockList 中的每个 ASTGCN_block，将输入 x 依次通过这些块
        for block in self.BlockList:
            x = block(x)

        # 通过最终的卷积层 self.final_conv 处理输入 x，然后选择最后一个时间步的输出，并将其转置以得到最终的预测结果
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        # print(torch.mean(output))
        # 返回预测结果
        return output

# 加载网络结构
def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict,
               len_input, num_of_vertices):
    """
    
    :param DEVICE: 设备标识，用于指定模型运行在CPU或GPU上。
    :param nb_block: 模型中的图卷积块的数量。
    :param in_channels: 输入特征的维度。
    :param K: Chebyshev多项式的阶数，用于图卷积核的构建。
    :param nb_chev_filter: Chebyshev滤波器的数量。
    :param nb_time_filter: 时间滤波器的数量。
    :param time_strides: 时间卷积的步长。
    :param adj_mx: 图的邻接矩阵，表示图中节点的连接关系。
    :param num_for_predict: 预测时考虑的节点数量。
    :param len_input: 输入序列的长度。
    :param num_of_vertices: 图中顶点的数量。
    :return: 返回创建的模型实例。

    """
    # 固定随机数种子以确保可重复性
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Chebyshev多项式作为卷积核
    # scaled_Laplacian(adj_mx) 函数计算图的缩放拉普拉斯矩阵 L_tilde
    L_tilde = scaled_Laplacian(adj_mx)
    # cheb_polynomial 函数用于生成基于 L_tilde 和多项式阶数 K 的Chebyshev多项式。然后，这些多项式被转换成PyTorch的 torch.FloatTensor 类型，并且发送到之前定义的 DEVICE 上
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    # design model实例化了一个 ASTGCN_submodule 对象，它是模型的主体，包含了图卷积和时间卷积的配置。
    model = ASTGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                             cheb_polynomials, num_for_predict, len_input, num_of_vertices)
    # 遍历模型的所有参数 p。对于多维的参数（通常是权重矩阵），使用Xavier均匀分布进行初始化；对于一维的参数（通常是偏置项），使用均匀分布进行初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    # 返回创建的模型实例
    return model
