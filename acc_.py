import os
import pandas as pd
import jiwer
import ast
import numpy as np
from scipy.optimize import linear_sum_assignment

# 1️⃣ 设定固定存储目录
output_dir = "result_3__acc"
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ 加载标准答案 data_b.csv
ref_df = pd.read_csv("data_b.csv", dtype={"ID": str})  # 确保 ID 为字符串
ref_df.rename(columns={"ID": "uttid", "Sentence": "reference"}, inplace=True)
reference_texts = ref_df["reference"].tolist()

if len(reference_texts) != 50:
    raise ValueError(f"❌ 参考文本数量错误，应为 50，当前为 {len(reference_texts)}")

# 3️⃣ 遍历 asr_results 目录中的所有 CSV 识别结果
asr_results_dir = "results_3_"
asr_files = [f for f in os.listdir(asr_results_dir) if f.endswith(".csv")]
if not asr_files:
    raise ValueError("❌ 没有找到任何 ASR 识别结果文件，退出。")

for asr_file in asr_files:
    sub_dir = os.path.splitext(asr_file)[0]  # 文件名（去掉 .csv）
    asr_csv_path = os.path.join(asr_results_dir, asr_file)
    asr_df = pd.read_csv(asr_csv_path, dtype={"uttid": str})

    if len(asr_df) != 50:
        raise ValueError(f"❌ {asr_file} 句子数量错误，应为 50，当前: {len(asr_df)}")

    # 4️⃣ 解析 ASR 结果，提取每一句的识别文本
    asr_texts = []
    uttid_list = asr_df["uttid"].tolist()  # 保存 uttid 顺序
    for _, row in asr_df.iterrows():
        raw_transcription = row["transcription"]
        try:
            if isinstance(raw_transcription, str):
                transcription_data = ast.literal_eval(raw_transcription)
                asr_text = transcription_data.get("text", "").strip()
                asr_texts.append(asr_text if asr_text else "空")
            else:
                asr_texts.append("空")
        except Exception as e:
            print(f"❌ 解析 {row['uttid']} 失败: {e}")
            asr_texts.append("空")

    if len(asr_texts) != 50:
        raise ValueError(f"❌ 解析出的 ASR 结果数量错误，应为 50，当前: {len(asr_texts)}")

    # 5️⃣ 预处理：去掉标点和多余空格
    transform = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces()])
    processed_ref = [transform(s.strip()) for s in reference_texts]
    processed_asr = [transform(s.strip()) for s in asr_texts]

    # 去除空格便于计算字符 CER（中文通常不用空格分词，这里直接移除空格）
    processed_ref_chars = [s.replace(" ", "") for s in processed_ref]
    processed_asr_chars = [s.replace(" ", "") for s in processed_asr]

    # 6️⃣ 构建 50x50 的代价矩阵（cost matrix），cost 为参考句与 ASR 句之间的 CER
    cost_matrix = np.zeros((50, 50))
    for i in range(50):
        for j in range(50):
            cer_val = jiwer.cer(processed_ref_chars[i], processed_asr_chars[j])
            cost_matrix[i, j] = cer_val if cer_val <= 1.0 else 1.0

    # 7️⃣ 用 Hungarian 算法求解最优分配问题
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind 对应参考句索引， col_ind 对应 ASR 句索引

    # 8️⃣ 根据匹配关系计算每对的 CER，并构建匹配结果
    matched_ref = [reference_texts[i] for i in row_ind]
    matched_asr = [asr_texts[j] for j in col_ind]
    cer_list = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]

    avg_cer = np.mean(cer_list) if cer_list else 0

    # 9️⃣ 生成 CER 结果文件
    # 保存匹配关系时，uttid 取 ASR 结果对应的 uttid（根据 col_ind 顺序排序）
    matched_uttid = [uttid_list[j] for j in col_ind]
    cer_csv = os.path.join(output_dir, f"{sub_dir}_cer.csv")
    compare_df = pd.DataFrame({
        "uttid": matched_uttid,
        "reference": matched_ref,
        "transcription": matched_asr,
        "CER": cer_list
    })
    compare_df.to_csv(cer_csv, index=False)

    # 🔟 输出最终 CER 结果
    print(f"📊 {sub_dir} 识别结果:")
    print(f"✅ 平均 字符错误率（CER）: {avg_cer:.2%}")
    print(f"📂 详细对比结果已保存到 {cer_csv}")

print("🎉 所有 ASR 结果处理完成！")