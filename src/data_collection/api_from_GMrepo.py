import os
import json
import logging
import requests
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_project_ids(file_path):
    with open(file_path, 'r') as file:
        return [line.split('\t')[0].strip() for line in file if line.strip()]

def get_taxonomic_profile(run_id, disease):
    url = 'https://gmrepo.humangut.info/api/getFullTaxonomicProfileByRunID'
    query = {"run_id": run_id}
    
    try:
        response = requests.post(url, data=json.dumps(query), timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # 获取数据
        run = data.get("run")
        species = pd.DataFrame(data.get("species"))
        genus = pd.DataFrame(data.get("genus"))
        
        # 创建安全的文件夹名
        safe_run_id = ''.join(c for c in run_id if c.isalnum() or c in ('-', '_'))
        
        # 修改输出目录路径
        output_dir = f"data/raw/{disease}/raw_data/{safe_run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "run_info.json"), 'w') as f:
            json.dump(run, f)
        
        species.to_csv(os.path.join(output_dir, "species_abundance.csv"), index=False)
        genus.to_csv(os.path.join(output_dir, "genus_abundance.csv"), index=False)
        
        logger.info(f"Successfully downloaded and saved taxonomic profile for {run_id}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download taxonomic profile for {run_id}: {str(e)}")
        return False

def process_disease(disease_path):
    disease = os.path.basename(disease_path)
    file_path = os.path.join(disease_path, f'{disease}.txt')
    
    try:
        run_ids = read_project_ids(file_path)
    except Exception as e:
        logger.error(f"Error reading project IDs for {disease}: {str(e)}")
        return 0, 0
    
    successful_downloads = 0
    failed_downloads = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_run_id = {executor.submit(get_taxonomic_profile, run_id, disease): run_id for run_id in run_ids}
        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            try:
                if future.result():
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            except Exception as exc:
                logger.error(f'{run_id} generated an exception: {exc}')
                failed_downloads += 1
    
    logger.info(f"{disease}: Successfully downloaded taxonomic profiles for {successful_downloads} runs")
    logger.info(f"{disease}: Failed to download taxonomic profiles for {failed_downloads} runs")
    return successful_downloads, failed_downloads

def main(target_disease=None):
    base_path = "data/raw"
    if target_disease:
        disease_path = os.path.join(base_path, target_disease)
        if not os.path.isdir(disease_path):
            logger.error(f"The specified disease directory {disease_path} does not exist.")
            return
        successful, failed = process_disease(disease_path)
    else:
        logger.error("Please specify a disease directory.")
        return
    
    logger.info(f"Total: Successfully downloaded taxonomic profiles for {successful} runs")
    logger.info(f"Total: Failed to download taxonomic profiles for {failed} runs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download taxonomic profiles from GMrepo for a specific disease.")
    parser.add_argument("disease", type=str, help="Specify the disease directory under 'data/raw' to process")
    args = parser.parse_args()
    
    # Sanitize the disease name to create a valid directory name
    safe_disease_name = ''.join(c for c in args.disease if c.isalnum() or c in ('-', '_', '&'))
    
    main(safe_disease_name)
