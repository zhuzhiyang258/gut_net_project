import optuna
import plotly
import os  

# 确保数据库文件路径正确
db_path = os.path.join("optuna_results", "optuna.db")
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file not found: {db_path}")

# 加载已保存的 study
storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
study = optuna.load_study(study_name="gutnet_optimization", storage=storage)


# 1. 优化历史
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# 2. 参数重要性
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# 3. 参数关系
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()

# 4. 超参数分布
fig = optuna.visualization.plot_slice(study)
fig.show()

# 5. 中间值分布
fig = optuna.visualization.plot_intermediate_values(study)
fig.show()

# 保存图表
output_file = 'optimization_results.html'
plotly.io.write_html(fig, file=output_file, auto_open=True)
print(f"Results saved to {output_file}")