import pandas as pd
import os

class DataCleaner:
    def __init__(self, input_directory, intermediate_directory, output_directory):
        self.input_directory = input_directory
        self.intermediate_directory = intermediate_directory
        self.output_directory = output_directory
        self.metadata_columns = ['loaded_uid', 'sex', 'host_age', 'country', 'BMI']

    def process_csv(self, file_path, output_directory, zero_threshold=0.4):
        df = pd.read_csv(file_path)
        zero_proportion = (df == 0).mean()
        columns_to_keep = zero_proportion[zero_proportion <= zero_threshold].index.tolist()
        df_filtered = df[columns_to_keep]
        
        file_name = os.path.basename(file_path)
        new_file_path = os.path.join(output_directory, file_name)
        df_filtered.to_csv(new_file_path, index=False)
        
        print(f"处理完成：{file_name}")
        print(f"原始列数：{df.shape[1]}，处理后列数：{df_filtered.shape[1]}")
        print(f"处理后的文件已保存为：{new_file_path}")
        print("--------------------")

    def process_directory(self, input_directory, output_directory):
        if not os.path.exists(input_directory):
            print(f"输入目录不存在：{input_directory}")
            return

        os.makedirs(output_directory, exist_ok=True)

        for filename in os.listdir(input_directory):
            if filename.endswith(".csv"):
                input_file_path = os.path.join(input_directory, filename)
                self.process_csv(input_file_path, output_directory)

    def get_common_columns(self, directory):
        all_columns = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(directory, filename))
                all_columns.append(set(df.columns))
        
        common_columns = set.intersection(*all_columns)
        return list(common_columns)

    def standardize_csv_files(self, input_directory, output_directory, common_columns):
        os.makedirs(output_directory, exist_ok=True)
        
        for filename in os.listdir(input_directory):
            if filename.endswith('.csv'):
                input_path = os.path.join(input_directory, filename)
                output_path = os.path.join(output_directory, filename)
                
                df = pd.read_csv(input_path)
                df_common = df[common_columns]
                
                # 调整列顺序
                metadata_cols = [col for col in self.metadata_columns if col in df_common.columns]
                other_cols = [col for col in df_common.columns if col not in self.metadata_columns]
                new_order = metadata_cols + other_cols
                df_reordered = df_common[new_order]
                
                # 添加 label 列
                label = os.path.splitext(filename)[0]
                df_reordered['label'] = label
                
                df_reordered.to_csv(output_path, index=False)
                
                print(f"标准化完成: {filename}")
                print(f"原始列数: {df.shape[1]}, 标准化后列数: {df_reordered.shape[1]}")

    def run(self):
        # 第一步：处理原始数据，去除零值过多的列
        self.process_directory(self.input_directory, self.intermediate_directory)

        # 第二步：找出共有列并标准化
        common_columns = self.get_common_columns(self.intermediate_directory)
        print(f"共有列数: {len(common_columns)}")
        self.standardize_csv_files(self.intermediate_directory, self.output_directory, common_columns)

if __name__ == "__main__":
    input_directory = "data/processed"
    intermediate_directory = "data/cleared"
    output_directory = "data/standardized"

    cleaner = DataCleaner(input_directory, intermediate_directory, output_directory)
    cleaner.run()
