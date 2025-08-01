# 动手学习agent



## 1 配置大模型环境

### 1-1 基础python环境

- 创建正确conda环境：`conda create -n LLM python=3.10 -y`
- 运行`nvidia-smi`查询显卡配置，发现`Driver Version: 576.88`和` CUDA Version: 12.9`
- 去官网安装对应版本的CUDA Toolkit，下载12.9版本，选择符合本机的正确配置，安装。重启命令行，执行`nvcc -V`有版本信息，说明CUDA Toolkit安装成功（也可以在conda环境下直接安装CUDA）
- pip安装必要的库，注意安装torch需要去官网，找到对应的stable版本，复制官网推荐的下载命令进行安装，直接pip install可能会安装CPU版本
- `pip install transformers accelerate sentencepiece`
- 运行`env_test.py`

### 1-2 `Huggingface`开源大模型部署 - transformers库使用

- transformers库提供了`huggingface`的很多接口，便于通过命令在第一次部署的时候自动下载开源大模型并部署，集成了一系列功能
- datasets库提供了大量开源数据集，可以直接import调用
- 一般通过`pipline`实例化一个大模型对象，然后直接调用这个对象进行回应
- 一系列基本常用功能：
  - 设置库纯离线运行，只调用本地有的大模型 ： HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
  - 部署大模型的三种方式：
    - 在线调用开源预训练大模型
    - 自定义随机初始化的大模型
    - 读取本地存储的大模型
  - 自定义训练步骤
  - 保存当前训练好的模型 / 保存下载的模型，到指定的本地位置
- `hf cache scan`用以显示huggingface占用的本地存储
- `hf cache delete`用以删除下载的模型
- 有时直接使用pipline调用某个大模型会出问题，这是因为开发者没有正确编写json，可以手动调用



## 2 项目选题

### 2-1 选题考虑

第一个上手实操的项目。限制条件如下：

- 硬件：可行性验证设备是4060Ti 8GB显存，最终可以用32GB显存8卡A100服务器运行程序，倾向于完全本地运行
- 希望学到东西：一个担忧是如果调用一些成熟的项目，下载下来一键运行，那么其实没有从中学会任何东西，更具体地说就是对于一个新的任务仍然不知道该怎么去使用现有的开源模型和数据库做一个多agent系统完成希望完成的任务，因此最重要的是使用最基本的单元，比如huggingface上就能找到的开源大模型和开源数据库，用最通用的方法去构建基本组件，完成想要完成的事情，而不是仅仅学会一键调用别人做出的项目。
- 希望练习MCP协议使用
- 希望学会：如何针对一个需求和给定的数据集，自己构建和链接multi-agent的架构，并自己从base模型的基础上开始进行训练或微调，来实现我的功能
- 关于项目任务：
  - 最好能够复刻一个最近出现的可复刻的领域内文章，有清晰的教程可以follow
  - 下一阶段想做一个解决数学推导的双agent模型，一个agent用来生成数学命题的推导和证明过程，另一个agent负责一步一步检验逻辑正确性或数值检验
  - 现阶段还是做一个已有项目更适合学习

训练微调逻辑：

- 实现multi-agent进行训练或微调，从低到高有几个层级：
  - 完全不调整大模型，仅仅简单地将几个LLM用写好的prompt和MCP等协议连接起来，在商业应用开发中这是成本最低的方式
  - 对特定领域微调base模型，比如通过RAG或RLHF、利用QLoRA等增强某个agent的专业知识、特定场景推理能力等，单独针对某个LLM进行微调训练；本质上只是LLM的训练微调
  - 多个agent联合训练或微调，比如MARTI
- 我们首先采用比较简单的方式，只进行提示词连接LLM，并使用LoRA进行微调

### 2-2 项目设计

首先实现一个RAG，这也是一种非常简单的multi-agent的场景。先用来熟悉agent训练

任务：问答，解决可能刁钻的中文事实问题

数据：huggingface上的数据集OpenStellarTeam/Chinese-SimpleQA

各级消融实验模型架构：

- A0：直接使用LLM接收提问并回答
- A1：使用LLM，用MCP协议调用数据库中的知识，输出答案。
  - 数据流：
    - 1 使用qdrant-find-memories进行匹配查找，找出数据库中最符合的top-k个知识
    - 2 根据查到的知识合并上下文，传给LLM
    - 3 LLM回答问题
- A2：双agent，使用一个agent筛选排序，用另一个agent进行回答
  - 数据流：
    - 1 使用向量相似度匹配查找数据库最符合的top-k个知识
    - 2 用一个agent判断这些知识与问题的相关性，去除不相关的，判断是否需要进一步查找，循环查找知识，最终合并为prompt
    - 3 根据prompt回答问题


