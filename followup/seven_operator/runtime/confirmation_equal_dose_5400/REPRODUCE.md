# 固定阶段剂量核验的重建与复跑

这六条轨迹与主确认共享六个任务源；不是另外六个独立初始化。原首例两个阶段选择1/2块后设计本项补验，所有新输入和代码在任何附加训练与新查询前冻结。原注册、结果与失败记录保持不变。`PROTOCOL.md` 是科学协议，`REGISTRATION.json` 列出六例及输入身份；结果是否通过只以全部科学与原始输出审计为准。

## 重建已存结果

先按 `confirmation_factored_5300/REPRODUCE.md` 将主确认六份完整档案还原到 `/root/simods_next`。再将本项六份完整档案和 `fixed_dose_upload.tar.gz` 中的公共确认目录还原到相同根目录。主确认首例的完整档案使用导出键r5311，但根案例目录仍是 `confirmation_world1_seed1_5301`。

公共根目录中需有原主确认复跑包内的 `audit_component_outputs.py`、`analyze_confirmation_transitions.py` 等审计工具，以及完全相同字节的基座/tokenizer。基座的官方revision及软件版本沿用主确认记录。对每例运行：

```sh
python /root/simods_next/confirmation_equal_dose_5400/audit_equal_dose_case.py --study /root/simods_next/equal_dose_world1_seed1_5401
```

其余目录从注册文件读取。审计允许训练父权重位于本例或相应主确认源目录，逐项核对继承与复用的真实权重、训练日志和父权重身份，重建两阶段各2块/1204更新的连续谱系，计算被改写标签的算子/配对总暴露，重建实际token、停止、所有逐项目分数和统计文件。`CASE_RAW_AUDIT.json` 同时区分计算一致性与科学门槛通过。原始输出重建不重新训练或查询神经模型。

## 从已完成主确认源重新训练

在包含完整主确认输出的新工作根目录中解压 `fixed_dose_upload.tar.gz`。附加案例目录应只有冻结输入，不能含以前的权重、输出或完成标记。必须先为所有原主确认案例产生合格的原始审计记录。然后执行冻结的常规串行调度器：

```sh
cd /root/simods_next/confirmation_equal_dose_5400
python -u run_suite.py
```

每例从原选定的acquired-three开始。第一阶段固定2块，第二阶段从该两块终点继续固定2块；干净轨迹具有相同累计预算。已有块仅在训练脚本、训练数据、分支/阶段/块编号、种子和父权重均符合固定路径时复用，并写入 `REUSED_COMPONENTS.json`，否则在新目录按冻结配方计算。保留原始绝对路径的runtime，不改写训练历史。两条轨迹的全部端点在新完整程序评测前固定。门槛失败保留，不追加训练。

实际租用会话曾让正在运行的原第二例自然结束，先执行附加首例，再恢复其余原确认，最后执行附加其余五例。这只是GPU调度顺序，未改变任何案例配方；记录为 `PRIORITY_SCHEDULING_RECORD.json`。常规复跑不需要暂停其它调度器，也不应同时启动常规和优先调度。

若需生成审计后的云端完整档案，科学运行结束后可执行 `postprocess_suite.py`，其需要公共根目录的 `stage_terminal_archives.py`。归档工具本身不决定科学通过。

## 统计与设计范围

每项评测的独立单位仍为共享根的二值项目簇。主确认和附加核验分别拥有自己的同时置信保证，不把两项的区间拼成一个未经校正的联合95%带。总训练初始化数仍为6。每阶段的改写总暴露匹配，不表示两个连续端点的累计历史相同。新鲜测试银行、预算和数值门槛均不能根据结果修改。

S7最新共享私有图设计的CPU计算位于 `theory_experiment_closure_4800/shared_graph_rank_design.py` / `SHARED_GRAPH_RANK_DESIGN.json`；其依赖同目录的 `verify_balanced_rank_family.py` 中整数消元函数。主文实验区remark与 `SIMODS20261022/e6_shared_graph_design.tex` 给出构造和计数证明。它验证任意预设连接秩的有限复形构造，不是新增神经模型实验。旧 `shared_pool_rank_design.py` 及推导/JSON作为有效的中间方案保留，未覆盖原注册文件。
