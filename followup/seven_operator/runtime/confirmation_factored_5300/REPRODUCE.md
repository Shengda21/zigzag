# 六案例确认的重建与复跑

本目录的 `REGISTRATION.json`、`PROTOCOL.md` 和六个案例的输入清单在首次确认训练前冻结。完成状态和科学通过状态不同；读取每例 `FINAL_RESULT.json` / `CONFIRMATION_FAILURE.json`，并同时读取 `CASE_RAW_AUDIT.json`。不得只保留通过的案例。

部署修复记录：原输入清单漏列已有的 `prompts.py`，首例完整程序脚本因此在导入时中断，未产生模型查询。该文件与训练数据生成模板、两项开发实验完全相同，并与六例全部冻结数值提示逐条核对一致。补齐发生在六例首次完整程序查询前，原注册、清单、权重、训练预算和判据均未改动。见 `TECHNICAL_DEPENDENCY_REPAIR.json`。首例中断前档案使用导出键 r5301；从同一组选定权重恢复后的完整档案使用导出键 r5311，案例身份始终为 r5301。

## 原始结果重建

六份案例的完整档案逐个存储。档案中保留注册输入、每次训练的适配器、训练日志、组件原始 token、完整程序原始 token、类别概率、按项目结果和分析文件。`ARCHIVE_VERIFICATION.json` 记录 C 盘与项目目录两份无损档案的核验。归档后添加的本地文稿与图表不属于冻结输入。

在 Linux 下将案例按原目录名解压到 `/root/simods_next`，并将 `cloud_tools/audit_component_outputs.py`、`audit_confirmation_case.py`、`analyze_confirmation_transitions.py`、`audit_confirmation_initialization.py` 放入该目录。这样能直接核对原始训练谱系中保留的绝对路径。使用每例 `ENVIRONMENT.json` 所列的软件版本。通过 `SIMODS_BASE_MODEL` 指向与 `BASE_MODEL_SNAPSHOT.json` 一致的 Qwen 基座/tokenizer；审计只使用 tokenizer 和已存输出，不调用 GPU 生成。

对每一个案例执行：

```sh
python /root/simods_next/audit_confirmation_case.py --study /root/simods_next/confirmation_world1_seed1_5301
```

将末尾目录替换为注册文件内六个 `directory` 值，全部运行。该审计依次重建组件选择、核对父适配器与训练步数、从原始 token/概率重建自主程序轨迹、重算每项得分和两次可能查看的同时统计。其通过只表示计算一致性；科学门槛失败仍为失败。

初始化审计另外重算 8,000 条策略测试的所有类别与概率，检查 128 条更新记录，并利用从未参与训练的注意力矩阵第 9–16 行核对注册随机初始化。它只重建初始随机数，不重新训练。数值模块的实际训练种子与冻结训练代码的角色、阶段、块偏移逐项比对。该附加记录保存在公共确认目录的 `POLICY_SEED_AUDITS` 中。

## 重新训练

使用本目录的 `reproducibility_inputs_with_dependency.tar.gz` 在新的空运行根目录解压，其内容逐文件核验记录是 `INPUT_ARCHIVE_VERIFICATION.json`。不要使用缺少提示依赖的最初上传包，也不要在已完成案例目录里直接执行训练。输入子集是每例 `manifest.json` 的 `files` 字段列出的文件，加上 `manifest.json` 本身，以及部署修复补齐的 `prompts.py`、`DEPENDENCY_REPAIR.json`；不包含旧的权重、组件/程序输出和完成标记。保持案例目录名、数据、注册种子、选择规则、预算和统计规则不变。修复没有改写原清单，新依赖的内容身份单独记录。

核对七个基座/tokenizer 文件与 `BASE_MODEL_SNAPSHOT.json`。已由官方模型注册表及实际文件复核：它们对应 `Qwen/Qwen2.5-1.5B-Instruct` 的 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。权重的 LFS 内容标识与实验快照一致，六个配置/tokenizer 文件的实际字节同时匹配注册表 Git 对象及实验快照；记录见 `BASE_PUBLIC_REVISION_CHECK.json`。完整模型基座属于外部依赖，实验所学全部权重保留在案例档案中。默认模型路径为 `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`，也可设置 `SIMODS_BASE_MODEL` 为完全相同字节的本地目录。

将 `REGISTRATION.json`、`run_six.py` 放在与六案例平级的 `confirmation_factored_5300` 子目录中，执行：

```sh
cd /root/simods_next/confirmation_factored_5300
python -u run_six.py
```

此命令按注册顺序串行执行全部六例。每例的七个端点在完整程序评测前选定；不依赖高任务分数重新选权重。首次查看为 256 根簇，只有点判据全过而同时界不足才增加预先冻结的另 256 根簇。失败种子、组件和所有已产生的查询均保留。不同设备的数值结果可能不同，复跑以同一科学门槛重新判定。

## 数据与统计范围

一个独立样本是共享根、两条程序合取的二值项目簇。各面、各检查点和每条轨迹中的步骤可以相关；不能把步骤数当样本量。置信保证针对训练结束后的固定模型与注册的抽样生成器，不是训练种子总体区间。两个固定规则世界各有三个任务训练初始化，不等于六个独立规则世界。

原单体模型、局部标记、换键及密集换键的失败另行保留。当前受控模型是已知神经执行分解的实现，不能将其通过写成原单次调用模型已获组合能力，或将架构分解本身称为新算法。

## 后续固定剂量核验

主确认首例的两个阶段分别选择1/2块，另有 `confirmation_equal_dose_5400` 的固定每阶段2块核验。它继承同一六源，使用新测试簇，不能计作另外六个独立初始化。原主确认的结果和注册保持独立。见该目录的 `PROTOCOL.md` 与 `REPRODUCE.md`；完整结果必须分别重建两项研究。
