cnn text 加噪和裁剪参数

checkpoints 保存checkpoint
* cacheckpoint 先裁剪后加噪的检查点。
    训练次数200，正态分布噪声参数为0.5,裁剪阈值1.1 dropout=0
* accheckpoint 先加噪后裁剪
    训练次数200，正态分布噪声参数为0.5,裁剪阈值1.1 dropout=0
* 正常未加噪的checkpoint训练60次
    
    
result保存输出的结果文件
*   cnnclipadd0.5为使用正太分布的随机数加噪先裁剪后加噪
    加噪参数为0.5,学习率：.15,batch_siza=256 训练次数：200
    裁剪阈值：1.1;dropout=0
*   cnnaddclip0.5cnnclipadd0.5为使用正太分布的随机书先加噪后裁剪
    加噪参数为0.5,学习率：.15,batch_siza=256 训练次数：200
    裁剪阈值：1.1;dropout=0