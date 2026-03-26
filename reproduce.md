# 复现mobile-aloha工作

本文基于[mobile-aloha的开源代码](https://github.com/MarkFzp/act-plus-plus)复现部分工作，按照从浅到深，包含三大部分的内容：
- Step1：单纯的跑通代码
- Step2：ACT算法和实现细节讲解，并且最大化的复现论文中的实验结果
- Step3：关于该论文的一些思考

**注意：本文仅仅包含mobile-aloha软件方面的结果复现，也就说基于仿真数据完成策略的学习，不涉及任何硬件和实际平台的内容。**

# 跑通代码
首先需要下载仓库源代码，链接：https://github.com/MarkFzp/act-plus-plus。
由于源代码有一些小错误，或者交代不清，因此我对它做了一些修改。参考代码仓库[huiwenzhang/act++](https://github.com/huiwenzhang/act-plus-plus)。嫌麻烦的朋友，也可以直接pull这个仓库代码。相比于源仓库主要的改动有：
- 增加了`requirements.txt`编译依赖
- 修复了部分运行错误
- 增加了部分shell脚本，方便测试，位于`scripts`文件下
- 如果需要可视化或者测试实际的数据，还需要用到[aloha-hardware](https://github.com/MarkFzp/mobile-aloha) 中的部分代码，主要是`aloha_scripts`文件下的一些脚本，因此本仓库直接将该文件夹复制到了根目录下。

*下面的复现过程都是基于改动后的仓库，如果发现文件不对，请确认是否拉取了正确的代码。*

## 依赖安装
仓库中部分依赖阐述不是很清楚，作者后续做了补充。我新建了一个requirements.txt文件，内容如下：

```
torchvision
torch
pyquaternion
pyyaml
rospkg
pexpect
mujoco==2.3.7
dm_control==1.0.14
opencv-python
matplotlib
einops
packaging
h5py
ipython
wandb
robomimic
diffusers
```

使用者可以通过`pip install -r requirements.txt`安装相关依赖。

其他部分代码中有些错误，大家可参照`https://github.com/MarkFzp/act-plus-plus/issues`的内容查阅。本代码已经做了修改，所以也可以直接pull本仓库的代码。

另外需要注意的是默认代码使用了wandb来log和可视化训练过程。如果也想自己可视化训练过程，需要修改wandb的用户名和key，具体修改的代码位于`imitate_episodes.py`的`main`函数，如下：
```python
  if not is_eval:
    wandb.init(project="mobile-aloha2", reinit=True, entity="moma", name=expr_name)
    wandb.config.update(config)
```
moma是自己在wandb创建的工作组，mobile-aloha2是在工作组下创建的项目。读者需要将它替换成自己wandb账号的项目。如果没有账号，可以直接把这几行代码注释掉运行。但是建议大家自己注册一个wandb账号。如果没有账号，需要注释掉所有wandb相关的代码。

## 数据集准备
数据集分为两类，分别是实际采集的数据和仿真数据。作者提供下载地址：
- 实际数据：https://drive.google.com/drive/folders/1FP5eakcxQrsHyiWBRDsMRvUfSxeykiDc
- 仿真数据：https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O

注意
- 仿真数据也可以自己生成，具体生成方法下面会介绍
- 如果直接使用实际数据进行训练会报错，提示需要安装关于机器人的一些ros包。这是因为在训练过程中会调用`eval_bc()`函数，该函数会使用学习的策略和实际机器人交互，通过交互的统计结果来评估策略的好坏。而我们没有连接真实物理世界的机器人，因此没办法运行这部分代码。如果单纯的想要使用实际数据来训练策略，可以将eval_bc代码注释掉。如下：

```python
# evaluation
if (step > 0) and (step % eval_every == 0):
    # first save then eval
    ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save(policy.serialize(), ckpt_path)
    # 注释掉下面两行
    # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
    # wandb.log({'success': success}, step=step)
```
> mobile aloha论文中的一个突出发现是使用co-train的训练方式可以大大提升任务成功率。虽然没有实际的机器人，也可以通过上面介绍的方式来验证co-train的效果。具体验证步骤是： 1）注释训练代码中eval_bc部分代码；2）将co-train的数据和仿真数据混合，训练任务，得到策略1；3）单纯只使用仿真数据，训练一个策略2；4）通过比较策略1和策略2的效果，就可以验证co-train是否有效。

**非常重要的一点，无论是后面训练还是保存数据，都需要指定自己的数据目录.**
该目录在文件`constants.py`或者`aloha_scripts/constants.py`里面设置。下载下来解压的数据和后面生成的数据都仿真这个文件夹下面。修改路径代码如下：
```python
### Task parameters
# DATA_DIR = '/home/zfu/interbotix_ws/src/act/data' if os.getlogin() == 'zfu' else '/scr/tonyzhao/datasets'
DATA_DIR = '/home/alvin/data/aloha/'  # 将该目录替换成自己的目录
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
---
}
```


下载解压后的数据目录如下：
![](docs/dataset_dir.png)


### 数据可视化
为了可视化数据集，可以使用`scripts/visulize_eps.sh`脚本。用法：
```
# 可视化实际采集数据
./scripts/visualize_eps.sh [task_name] [episode_idx] real
# 可视化仿真数据
./scripts/visualize_eps.sh [task_name] [episode_idx]
```
- task_name：表示任务的名字，比如`sim_transfer_cube`
- episode_idx：表示示教demo的序号
- real：表示要可视化仿真数据还是下载的实际数据


注意对于仿真数据集和实际采集的数据集，要使用不同的脚本来可视化。该仓库中提供的可视化python脚本只适用于仿真数据，也就是`sim_insertation_scripted`和`sim_transfer_cube_scripted`任务。对于下载的实际示教的数据，需要使用`visualize_episodes_real.py`脚本。具体原因参考：https://github.com/MarkFzp/act-plus-plus/issues/16。
实际上对比两个脚本的代码可以发现，对于实际采集的视频数据，经过了压缩处理，因此在加载数据时需要进行解压缩。然后再保存成视频。如果不经过压缩，会发现加载的视频图像的维度是错误，因此在执行concanate操作时，会出错。

可视化仿真数据实例：https://github.com/huiwenzhang/act-plus-plus/blob/main/docs/sim_insertion_scripted_episode_30_video.mp4

可视化实际数据示例：https://github.com/huiwenzhang/act-plus-plus/blob/main/docs/mobile_aloha_wash_pan_episode_20_video.mp4




### 生成仿真数据
使用方法：`./scripts collect_eps.sh [task_name]`。注意这里的task_name不能随便给，要看在`constants.py`中设置了哪些任务的配置，才能使用对应的任务名。

**生成数据的保存位置通过shell中的dataset_dir指定，使用者需要将其替换成自己本地的目录。**
最好和`constants.py`中的设置保持一致。

作者在项目中提供了两个仿真任务，分别是`sim_inertion_scripted`和`sim_transfer_cube_scripted`。生成的数据会保存成hdf5的格式。

- sim_inertion_scripted：该任务实现了一个插孔任务，双臂协作完成插孔配合。
![](docs/insert.png)

- sim_transfer_cube_scripted：该任务中一个机械臂需要拾取随机放置的方块，并递给另一个机械臂。
![](docs/transfer.png)



## 算法训练
使用方法：`./scripts/train.sh [task_name]`。
需要提前准备好对应任务的数据。脚本说明：
```sh
python3 imitate_episodes.py --task_name $1 --ckpt_dir ckpt/$1/ \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 \
    --batch_size 4 --dim_feedforward 3200  --num_steps 20000 --lr 1e-5 --seed 0 
```
部分参数说明：
 - task_name: 任务名称
 - ckpt_dir：保存网络权重文件的位置，默认在仓库根目录的`ckpt/`目录下
 - policy_class：采用的模仿学习算法，默认ACT，其他的暂时没有测试
 - batch_size：训练时batch size大小，实验中发现大于4后，出现`cuda memory error`，应该是显存大小不够，可以根据自己的GPU情况修改
 - num_steps：训练的步数，本实验中设置为20000步。如果训练的策略评估不好，可以适当增加训练的步数

 训练过程中可视化结果如下：
 ![](docs/train_sim.png)

经过20000步训练，成功率依然比较低，但是大多情况下已经能完成任务。

## 算法评估
本实验分别对`sim_insertion_scripted`和`sim_transfer_cube_scripted`任务进行了学习。学习后的策略保存在`ckpt/sim_insertion_scripted`和`ckpt/sim_transfer_cube_scripted`文件夹。每间隔500步会保存一次策略，因此该文件夹下会保存很多权重文件。最优的策略和最后策略的命名为`policy_best.ckpt`和`policy_last.ckpt`。作者源代码中在评估策略时默认使用的是policy_last.ckpt文件，我将其改成了policy_best.ckpt。

为了可视化策略的效果，可以运行脚本`./scripts/eval.sh`。脚本中加了`--onscreen_render`参数是为了可视化任务执行的结果。如下所示：

- 插孔任务

![](docs/insert_test_res.png)

其实对于插孔任务，前三步的成功率达到了90%，只是最后插入后由于干涉等原因导致任务失败。如果增加训练步数应该可以解决这个问题。其次是任务的训练中没有用到力的信息，所以接触类的任务学习起来更难。

- 转移任务

![](docs/transfer_test_res.png)

transfer任务更加容易，成功率也更高。

两个任务的测试视频可以在`docs`文件夹，读者可以自行查看。



# 算法实现细节解析
mobile-aloha对的核心还是ACT算法。由于ACT本质上还是一种模仿学习方法。模仿学习顾名思义就是一个生手通过学习一个老手的行为，最终期望能复现老手甚至超越老手，所谓青出于蓝而胜于蓝。为了实现这个目的最naive的方法就是行为克隆（behavior clone, BC）。但BC没办法做到青出于蓝而胜于蓝，而且有误差累积等一些列问题。所以大家一直在找一种能够学习示教数据分布的方法。恰好GAN、VAE、以及最近的比较火的扩散模型等都是干这个事情的，因此很自然的就把这些模型引入到了模仿学习中。比如结合GAN的模仿学习方法GAIL，当初也很流行。本文则是用了VAE的架构，其实VAE和GAN也有很深的渊源，这个不深入介绍，参考资料很多，大家自行学习。

说回来，怎么使用VAE呢？整体就是如下的架构：
![](docs/vae.png)
VAE架构就三个东西：编码器、隐变量、解码器（生成器）。大家不要看到解码器里面有个transformer encoder和decoder就迷糊了。这个tansformer编码器和解码器就是VAE解码器的实现网络架构。

关于ACT算法的实现代码位于`detr/models/detr_vae.py`文件的*build*函数，代码如下：
```python
def build(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    # 创建vae中的解码器或者叫生成器，通过输入
    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        # encoder = build_transformer(args)
        # 创建VAE中的解码器
        encoder = build_encoder(args)

    # 负责将编码器解码器融合起来，并做一些编码器解码器的输入预处理工作
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vq=args.vq,
        vq_class=args.vq_class,
        vq_dim=args.vq_dim,
        action_dim=args.action_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model
```
## 编码器
`build_encoder`函数负责创建编码器。编码器输出一个隐变量，是一个32维的高斯分布，encoder的架构如下：
![](docs/encoder.png)

实现代码位于`detr/models/detr_vae.py`的DETRVAE类的`encode`函数，文件中92行。代码如下：

```python
def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
    bs, _ = qpos.shape
    if self.encoder is None:
        latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
        latent_input = self.latent_out_proj(latent_sample)
        probs = binaries = mu = logvar = None
    else:
        # cvae encoder
        is_training = actions is not None # train or val
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            
            if self.vq:
                logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                probs = torch.softmax(logits, dim=-1)
                binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                latent_input = self.latent_out_proj(straigt_through)
                mu = logvar = None
            else:
                probs = binaries = None
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]
                latent_sample = reparametrize(mu, logvar)
                latent_input = self.latent_out_proj(latent_sample)

        else:
            mu = logvar = binaries = probs = None
            if self.vq:
                latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
            else:
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                latent_input = self.latent_out_proj(latent_sample)

    return latent_input, probs, binaries, mu, logvar
```
主要的代码看3行就行，分别是编码器的输入：
```python
encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) #
```
基于transformer的表征：
```python
encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
```
输出结果：
```python
latent_info = self.latent_proj(encoder_output)
```

## 解码器
在训练阶段解码器的输入包括当前关节信息、来自编码器的隐变量Z以及多个相机的RGB信息。通过transformer编码器和解码器，输出下一步预测的动作序列。实现代码对应于：
![](docs/decoder.png)

代码细节也就不说了，对照一下，就看的很清楚了。

## 关于推理
都知道对于VAE在推断的时候不需要编码器，对应的隐变量设置为标准的高斯分布，如下：
![](docs/inference.png)

红色框框中的输入值设为0。在代码中如何实现的呢？代码中有个参数叫`--no-encoder`，起初以为在evalation阶段把它设置成true，从而实现推理。最后发现不是的。其实推理和训练有一个区别是否给定了action。对于训练数据，有action信息，对于推理action是网络的输出，提前不知道，因此可以通过判定action是否为none来确定是出于训练阶段还是推理阶段。代码中是这样实现的：

![](docs/disable_encoder.png)

# 后续
本文只是花了一点时间，简单的对照代码和论文做了一个梳理。很多细节没有涉及到。比如

- co-train对策略的影响
- 论文中提出的Temporal Ensemble的效果
- 一些参数对结果的影响，比如action chunk中的chunk size，KL权重大小

有时间的话可以多跑跑实现去观察对比一下。其次对于这种基于图像学习动作的方法，我称之为*泛视觉私服*，我认为是很有价值的。在于：第一：提供了一种基于稠密反馈的任务执行方式，应该说调的好的话，比scripted policy的鲁棒性更强； 第二：确实有希望在low-cost或者使用低成本的传感器实现精细的操作；第三：提供了一种对于任务描述和表达的方式。其实对于很多任务我们可能没办法用数学模型来表达，更别说得到一个scripted的策略。但是对于这些工作也有很多疑问，比如：泛化性如何？上限在哪里？因为使用的是图像输入，如果相机的位置变化了，输入会发生很大的变化，策略是否能够overcome这种情况？还有论文中说到使用绝对的动作输出效果更好，但是从以往的论文来看大家更倾向于学习增量或者变化量，这样更鲁棒。对于接触操作，引入触觉或者力是否更有效？当前一个任务一个策略，是否可以学习一个策略适用多任务？还有这篇文章还是侧重于执行层面，应该也可以结合大模型，将任务规划也考虑进来。所有这些问题都可以再进一步深究。