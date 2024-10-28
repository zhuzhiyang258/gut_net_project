import pandas as pd
import json
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_genus_abundance(base_path, batch_size=1000):
    all_columns = set()
    total_files = sum([len(files) for r, d, files in os.walk(base_path) if 'genus_abundance.csv' in files])
    processed_files = 0

    # 第一次遍历，收集所有可能的菌种列
    for disease in os.listdir(base_path):
        disease_path = os.path.join(base_path, disease, 'raw_data')
        
        if os.path.isdir(disease_path):
            for sample_folder in os.listdir(disease_path):
                sample_path = os.path.join(disease_path, sample_folder)
                csv_file_path = os.path.join(sample_path, 'genus_abundance.csv')
                
                if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
                    try:
                        df = pd.read_csv(csv_file_path)
                        if 'scientific_name' in df.columns:
                            all_columns.update(df['scientific_name'].unique())
                        else:
                            print(f"Warning: 'scientific_name' column not found in {csv_file_path}.")
                    except pd.errors.EmptyDataError:
                        print(f"Warning: {csv_file_path} is empty and will be skipped.")

    # 第二次遍历，处理数据并添加所有菌种列
    for disease in os.listdir(base_path):
        disease_path = os.path.join(base_path, disease, 'raw_data')
        disease_data = []
        
        if os.path.isdir(disease_path):
            sample_folders = os.listdir(disease_path)
            for i, sample_folder in enumerate(tqdm(sample_folders, desc=f"Processing {disease}")):
                sample_path = os.path.join(disease_path, sample_folder)
                csv_file_path = os.path.join(sample_path, 'genus_abundance.csv')
                json_file_path = os.path.join(sample_path, 'run_info.json')
                
                processed_files += 1
                
                if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
                    try:
                        df = pd.read_csv(csv_file_path)
                        if 'scientific_name' not in df.columns or 'loaded_uid' not in df.columns:
                            logging.warning(f"Required columns not found in {csv_file_path}. Columns: {df.columns.tolist()}")
                            continue
                        
                        json_data = {}
                        if os.path.exists(json_file_path):
                            with open(json_file_path, 'r') as f:
                                json_data = json.load(f)
                        
                        pivot_df = df.pivot_table(index='loaded_uid', columns='scientific_name', values='relative_abundance', fill_value=0)
                        pivot_df.reset_index(inplace=True)
                        
                        # 创建一个包含所有列的 DataFrame
                        full_df = pivot_df.reindex(columns=pivot_df.columns.union(all_columns), fill_value=0)
                        
                        # 添加其他信息列
                        for key in ['host_age', 'sex', 'BMI', 'country']:
                            full_df[key] = json_data.get(key)
                        
                        disease_data.append(full_df)

                        # 每处理 batch_size 个文件就保存一次结果
                        if len(disease_data) >= batch_size:
                            save_batch(disease_data, disease, i // batch_size)
                            disease_data = []

                    except Exception as e:
                        logging.error(f"Error processing {csv_file_path}: {str(e)}")
                else:
                    logging.warning(f"{csv_file_path} does not exist or is empty.")
            
            # 保存剩余的数据
            if disease_data:
                save_batch(disease_data, disease, len(sample_folders) // batch_size)

    # 处理完所有数据后，合并批次文件
    merge_batches(base_path)

def save_batch(data, disease, batch_num):
    if data:
        combined_df = pd.concat(data, ignore_index=True)
        processed_dir = os.path.join('data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        output_file_path = os.path.join(processed_dir, f"{disease}_batch_{batch_num}.csv")
        combined_df.to_csv(output_file_path, index=False)
        logging.info(f"Saved batch {batch_num} for {disease}")

def merge_batches(base_path):
    processed_dir = os.path.join('data', 'processed')
    merged_data = {}

    logging.info("开始合并批次文件...")

    # 遍历processed目录中的所有CSV文件
    for filename in tqdm(os.listdir(processed_dir), desc="合并文件"):
        if filename.endswith('.csv'):
            disease = filename.split('_batch_')[0]
            file_path = os.path.join(processed_dir, filename)
            
            try:
                df = pd.read_csv(file_path)
                if disease not in merged_data:
                    merged_data[disease] = []
                merged_data[disease].append(df)
            except Exception as e:
                logging.error(f"处理文件 {filename} 时出错: {str(e)}")

    # 对每种疾病的数据进行合并
    for disease, dataframes in merged_data.items():
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)
            output_path = os.path.join(processed_dir, f"{disease}_combined.csv")
            combined_df.to_csv(output_path, index=False)
            logging.info(f"已创建合并文件: {output_path}")

            # 删除原始的批次文件
            for filename in os.listdir(processed_dir):
                if filename.startswith(f"{disease}_batch_") and filename.endswith('.csv'):
                    os.remove(os.path.join(processed_dir, filename))
                    logging.info(f"已删除批次文件: {filename}")

        except Exception as e:
            logging.error(f"合并 {disease} 数据时出错: {str(e)}")

    logging.info("批次文件合并完成，原始批次文件已删除。")

# 使用函数处理所有数据
base_path = 'data/raw'
process_genus_abundance(base_path)
