import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # 0,1,2,3,4,5,6,7
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '../')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
torch.backends.cudnn.enabled = False

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', 
                correction_net_name=None, correction_net_param=None, subject_name=None, 
                sequence=None, debug=0, threads=0, num_gpus=8, is_ttt=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name, subject_name=subject_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [
        Tracker(name=tracker_name, parameter_name=tracker_param, dataset_name=dataset_name, 
                correction_net_name=correction_net_name, correction_net_parameter_name=correction_net_param, 
                run_id=run_id)
    ]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='siamfc', help='Name of tracking method [seqtrack, stark_st, hiptrack, siamfc].')
    parser.add_argument('--tracker_param', type=str, default='siamfc', help='Name of config file [seqtrack_b256, stark_st2, hiptrack, siamfc].')
    parser.add_argument('--correction_net_name', type=str, default='correction_net', help='Name of tracking method [correction_net].')
    parser.add_argument('--correction_net_param', type=str, default='correction_net_l384', help='Name of config file [correction_net_l384].')
    parser.add_argument('--runid', type=int, default=5, help='The run id.')
    parser.add_argument('--subject_name', type=str, default='yinxiaozhe', 
                        help="Name of subject ['none', 'fanyanlong', 'liweilong', 'lupengyi', 'yinxiaozhe', 'liqing', 'zhaojiaxin', 'wuhaitao', 'chenyuan', 'duboai', 'liupengyu', 'zhoujiahao', 'fanhua', 'jiashuyu', 'renwenhao', 'huangyixiong',]. 'none' also means no corrector.")
    parser.add_argument('--dataset_name', type=str, default='nfs', help='Name of dataset (otb, nfs, uav, got10k_test, lasot, trackingnet, lasot_extension_subset, tnl2k).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--is_ttt', type=int, default=0)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, 
                args.correction_net_name, args.correction_net_param, args.subject_name, 
                seq_name, args.debug, args.threads, num_gpus=args.num_gpus, is_ttt=args.is_ttt)


if __name__ == '__main__':
    main()
    print('test over')

'''
001. lwh 
修改sampler train-runid = 6 眼动数据使用yinxiaozhe
seqtrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/seqtrack/seqtrack_b256_006/CorrectionNet_ep0005.pth.tar" 
hiptrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/hiptrack/hiptrack_006/CorrectionNet_ep0005.pth.tar"
stark_st
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/stark_st/stark_st2_006/CorrectionNet_ep0005.pth.tar"
siamfc
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/siamfc/siamfc_006/CorrectionNet_ep0005.pth.tar"

002. lwh 修改sampler train-runid = 6 眼动数据使用yinxiaozhe
seqtrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/seqtrack/seqtrack_b256_006/CorrectionNet_ep0010.pth.tar" 
hiptrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/hiptrack/hiptrack_006/CorrectionNet_ep0010.pth.tar"
stark_st
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/stark_st/stark_st2_006/CorrectionNet_ep0010.pth.tar"
siamfc
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/siamfc/siamfc_006/CorrectionNet_ep0010.pth.tar"

003. lwh 修改sampler train-runid = 6 眼动数据使用yinxiaozhe
seqtrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/seqtrack/seqtrack_b256_006/CorrectionNet_ep0015.pth.tar" 
hiptrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/hiptrack/hiptrack_006/CorrectionNet_ep0015.pth.tar"
stark_st
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/stark_st/stark_st2_006/CorrectionNet_ep0015.pth.tar"
siamfc
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/siamfc/siamfc_006/CorrectionNet_ep0015.pth.tar"

004. lwh 修改sampler train-runid = 6 眼动数据使用yinxiaozhe
seqtrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/seqtrack/seqtrack_b256_006/CorrectionNet_ep0020.pth.tar" 
hiptrack
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/hiptrack/hiptrack_006/CorrectionNet_ep0020.pth.tar"
stark_st
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/stark_st/stark_st2_006/CorrectionNet_ep0020.pth.tar"
siamfc
训练checkpoints路径"/data/guohua/BeiJing/code/TrackCorrector_lwh/checkpoints/train/siamfc/siamfc_006/CorrectionNet_ep0020.pth.tar"

005. lwh 修改sampler train-runid = 6 
眼动数据使用fanyanlong, liweilong, lupengyi, yinxiaozhe
使用各个Tracker的05epoch的checkpoints
对三个数据集进行测试 LaSOT, UAV, NfS

006. 测试siamfc的各个epoch的性能 5-32.87, 10-32.88, 15-32.97, 20-32.96
'''

'''
runid
999: 原模型性能 LaSOT280=69.47 NFS100=68.97 UAV123=69.39
1: epoch 60 AUC=3.03
2: epoch 200 仅转换量纲为search_size `correct_seq.reshape(-1,7)[:, 0:-3] / (self.bins-1) * self.params.search_size` 不测了
3: epoch 200 仅修改纠偏后的输出，与原策略保持一致，但map_box_back是基于当前帧的state。AUC=19.91
4: epoch 200 基于3，但map_box_back是基于上一帧的state。AUC=64.00
5: epoch 200 基于4，眼动xy转换使用上一帧state。AUC=64.00
6: epoch 300 基于5。AUC=64.22
7: epoch 300 基于6，加入compare_intervals=int(self.update_intervals * 0.02)和dist_threshold（使用之前非学习的参数）。AUC=69.90
8: epoch 300 基于7，compare_intervals=int(self.update_intervals * 0.02)。AUC=71.03
9: epoch 500 基于8。AUC=70.59
10: 注释correction_model，确保原性能不出错。AUC=69.47
11: epoch 300 仅针对难样本0%-20% 450*0.02/3=3帧。AUC280=70.47 AUC49=18.96 AUC27=21.27
12: epoch 300 仅针对难样本0%-20% 450*0.02/3=3帧 dist_th=50。AUC49=20.58 AUC27=21.41
13: epoch 300 基于11，1帧。AUC280=70.49 AUC49=21.02 AUC27=20.86
14: epoch 300 基于8，单独跑一些样本，用于导出纠偏过程中的seqtrack预测结果
15: epoch 300 基于8，dist_th=20。AUC27=17.05
16: epoch 300 基于8，推理替换output_seq的xy。AUC27=22.27
17: training epoch 100 基于8，correction_net仅在0%-10%的难样本上使用epoch300的ckpt进行训练。AUC27=24.35
18: seqtrack fine-tuning epoch 100 基于17，seqtrack_net仅在0%-10%的难样本上使用epoch500的ckpt进行微调。AUC27=31.24
19: seqtrack fine-tuning epoch 300 基于18。AUC27=33.83
20: seqtrack fine-tuning epoch 300 基于19，纠偏网络的image_list的search改成根据眼动xy获取。AUC27=27.33
21: 基于20，模板不更新。AUC27=26.29
22: 基于21，纠偏后利用眼动数据的xy作综合判断。AUC27=51.53
23: 基于22，模板再次改成更新（现推理纠偏策略：纠偏search使用眼动xy得到）。AUC280=22.00 AUC27=51.18
24: 基于8和23，  AUC27
25: 基于23，原模型+100轮纠偏网络难样本上重新训练。 AUC280=72.42 AUC27=39.13
26: 原seqtrack模型微调50轮，不加入纠偏网络和推理纠偏策略。AUC280=38.43 AUC27=31.91
27: 基于26，加入纠偏网络，及23的推理纠偏策略。AUC280=45.57 AUC27=49.00
28: seqtrack在0-20%上微调40epoch，和基于23，纠偏网络使用全样本训练epoch300。AUC27=47.05 AUC280=70.49
29: seqtrack在0-10%+每个类随机两个样本（除0-10已包含的类）上微调40epoch，和基于23，纠偏网络使用全样本训练epoch300。AUC27=44.24 AUC280=70.49
30: 基于17，原seqtrack+纠偏在0-10%难样本上训练200epoch。AUC27=39.30
31: 基于30，200epoch -> 300epoch 。AUC27=39.84
32: 基于31，300epoch -> 500epoch 。AUC27=40.04
33: 基于28，纠偏网络改为 0-10%样本上重新训练100epoch。AUC27=47.08
34: 基于29，纠偏网络改为 0-10%样本上重新训练100epoch。AUC27=44.01
35: seqtrack基于29的40epoch，纠偏网络为lwh基于困难帧训练的，推理策略为23。AUC280=70.72
36: seqtrack使用原模型，纠偏网络为lwh基于困难帧训练的，推理策略为23。AUC27=38.91
37: seqtrack使用原模型，纠偏网络为lwh基于困难帧训练的，推理策略为23，模板更新加入眼动和预测值的距离框内判断（否则复制第一帧）。AUC27=39.04
38: 基于36，调整非学习超参数，compare_intervals=0.02->0.2。AUC27=33.70
39: 基于36，调整非学习超参数，dist_threshold=100->50。AUC27=36.79
40: 当连续10帧眼动数据和预测结果不重叠时，利用纠偏网络校正结果，100像素距离阈值，修改纠偏网络输入search裁剪大小2倍。AUC27=41.23(原seqtrack性能18.53)，AUC280=73.72
41: 应用新计算的正则化的平均距离作为距离阈值，基于40。AUC27=39.13
42: 基于41，缩放平均距离1.2倍，连续8帧。AUC27=38.93，AUC280=72.58
43: 基于41，缩放平均距离1.2倍，连续13帧。AUC27=39.66，AUC280=73.12
44: 基于43，纠偏网络改为全训练样本训练300轮的，测试重新挑选的20个难样本。AUC20_ori=6.20 AUC20=38.19
45: 基于44，纠偏网络改为lwh训练的（错误帧选择(眼动和预测值的IOU<0.2)-search扩大1.2倍）AUC20=35.29 AUC280=69.87
46: 基于44，纠偏网络改为lwh训练的（错误帧选择(真值和预测值的IOU<0.2)-search扩大1.2倍）AUC20=35.77 AUC280=70.00
47: 基于42，纠偏网络使用lwh训练的（错误帧选择(真值和预测值的IOU<0.2)） AUC20=37.39 AUC280=72.46
48: 基于47和43的超参数。AUC20=39.78 AUC280=73.04
49: lwh的trainingid=6（训练正确的纠偏网络），每10帧（10%比例）应用一次眼动数据，缩放平均距离1.2倍。AUC280=8.28
50: lwh的trainingid=6（训练正确的纠偏网络），每4帧（10%比例）应用一次眼动数据，缩放平均距离1.2倍。AUC280=68.04
51: lwh的trainingid=6（训练正确的纠偏网络），每2帧（10%比例）应用一次眼动数据，缩放平均距离1.2倍。AUC280=67.83
52: /data/guohua/BeiJing/code/VideoX/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_wts_1_9/ AUC280=69.80
53: /data/guohua/BeiJing/code/VideoX/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_wts_2_8/ AUC280=69.77
54: /data/guohua/BeiJing/code/VideoX/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_wts_3_7/ AUC280=70.83
55: /data/guohua/BeiJing/code/VideoX/SeqTrack/test/tracking_results/seqtrack/seqtrack_b256_wts_5_5/ AUC280=71.26
56: 使用l384规格纠偏网络10epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.32
57: 使用l384规格纠偏网络20epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.50
58: 使用l384规格纠偏网络30epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.79
59: 使用l384规格纠偏网络40epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.58
60: 使用l384规格纠偏网络50epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.61
61: 使用l384规格纠偏网络60epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.55
62: 使用l384规格纠偏网络70epoch的ckpt，测试时不放大search，连续13帧，眼动距离阈值缩放1.2倍。AUC280=72.27
63: 使用l384规格纠偏网络5epoch的ckpt，训练时将target_seqs改为使用l384规格计算损失，其他与62保持一致。AUC280=72.55
64: 使用l384规格纠偏网络20epoch的ckpt，训练时将target_seqs改为使用l384规格计算损失，其他与62保持一致。AUC280=72.44
65: 使用l384规格纠偏网络40epoch的ckpt，训练时将target_seqs改为使用l384规格计算损失，其他与62保持一致。AUC280=72.54
66: 使用l384规格纠偏网络60epoch的ckpt，训练时将target_seqs改为使用l384规格计算损失，其他与62保持一致。AUC280=72.61
67: GT和seqtrack_b256预测值IoU小于0.5（错误帧）时，应用l384规格纠偏网络epoch60（training_id=003）。AUC280=67.64
68: 使用l384规格纠偏网络60epoch的ckpt，使用4个困难样本调试推理的两个超参数（hard_frame_count_threshold, dist_sacling_factor）
hard_frame_count_threshold=13 dist_sacling_factor=1.2 AUC4=29.98
hard_frame_count_threshold=4 dist_sacling_factor=1.2 AUC4=31.69
hard_frame_count_threshold=1 dist_sacling_factor=1.2 AUC4=29.98
hard_frame_count_threshold=8 dist_sacling_factor=1.2 AUC4=29.97
hard_frame_count_threshold=4 dist_sacling_factor=0.8 AUC4=30.04
hard_frame_count_threshold=4 dist_sacling_factor=1.0 AUC4=30.82
hard_frame_count_threshold=10 dist_sacling_factor=1.0 AUC4=30.24
hard_frame_count_threshold=20 dist_sacling_factor=1.2 AUC4=26.34
69: 使用l384规格纠偏网络60epoch的ckpt，hard_frame_count_threshold=4 dist_sacling_factor=1.2。AUC280=68.97 NFS100=68.88
70: 使用l384规格纠偏网络140epoch的ckpt，训练时将target_seqs改为使用l384规格计算损失，其他与62保持一致。AUC280=72.61
71: 使用l384规格纠偏网络100epoch的ckpt，训练时将target_seqs改为使用l384规格计算损失，其他与62保持一致。AUC280=72.03
72: 与71进行对比, 71是如果出现无效眼动数据, 则清零积累的错误帧; 72是不清零, 继续积累. 其他参数一致. AUC280=72.28
73: 8帧连续错误，缩放1.0，liweilong=0.0926距离。LASOT280=69.85 NFS100=66.88
74: 20帧连续错误，缩放1.0，liweilong=0.1019距离。LASOT280= NFS100=68.65 UAV123=65.37
75: 50帧连续错误，缩放1.0，liweilong=0.1019距离。LASOT280= NFS100=70.07 UAV123=66.37
76: 160帧连续错误，缩放1.0，liweilong=0.1019距离。LASOT280= NFS100= UAV123=
77: 100epoch的ckpt的l384纠偏，13帧连续，liweilong=0.1019距离。其他人的平均距离阈值先与liweilong保持一致。
LaSOT280 seqtrack= hiptrack= stark= siamfc=
UAV123 seqtrack= hiptrack= stark= siamfc=
NFS seqtrack= hiptrack= stark= siamfc=

假设场景
AI模型追踪目标，发生错误人为干预

AI模型连续30帧错误，人为干预开始采集眼动数据，采集30帧，pass。

累积30帧错误，人为干预一帧，矫正回来了，然后眼动连续判断30帧正确，停止干预，pass。

随机比例方案：
按照样本数量随机，原始seqtrack推理结果，加上纠偏的结果，计算按样本比例的目标追踪结果。75%原始，25%（10%、50%）纠偏，直接算AUC值。

10帧 4帧 2帧应用一帧眼动数据，进行纠偏

基于现在的策略，计算一共矫正了多少帧，

'''