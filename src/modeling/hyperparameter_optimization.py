import optuna
from trainer_without_attention import GUTNetTrainer, GUTNetConfig
import logging
import os
import sys
import torch
import multiprocessing
import joblib
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial):
    try:
        # 动态选择最空闲的 GPU
        gpu_id = select_gpu()
        torch.cuda.set_device(gpu_id)
        
        # 定义要优化的超参数
        config = GUTNetConfig(
            input_dim=11,
            num_classes=5,
            hidden_size=trial.suggest_int('hidden_size', 64, 256),
            num_hidden_layers=trial.suggest_int('num_hidden_layers', 2, 12),
            num_attention_heads=1,
            intermediate_size=trial.suggest_int('intermediate_size', 128, 1024),
            hidden_dropout_prob=trial.suggest_float('hidden_dropout_prob', 0.0, 0.5),
            attention_probs_dropout_prob=trial.suggest_float('attention_probs_dropout_prob', 0.0, 0.5),
            problem_type="single_label_classification",
            log_level="INFO"
        )

        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        data_path = 'data/standardized'
        trainer = GUTNetTrainer(config, data_path, batch_size=batch_size, lr=lr)
        trainer.model.to(f'cuda:{gpu_id}')

        # 训练模型
        best_val_accuracy = trainer.train(num_epochs=20, patience=5)

        return best_val_accuracy
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        return None

def select_gpu():
    # 简单的 GPU 选择策略，选择内存占用最少的 GPU
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        gpu_memory.append(torch.cuda.memory_allocated(i))
    return gpu_memory.index(min(gpu_memory))

def run_optimization(n_trials=100):
    # 创建SQLite数据库来存储优化结果
    storage = optuna.storages.RDBStorage(
        "sqlite:///optuna_results/optuna.db",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )
    
    study = optuna.create_study(
        direction='maximize', 
        storage=storage, 
        study_name="gutnet_optimization",
        load_if_exists=True
    )
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    # 增加并行度，每个 GPU 运行多个任务
    n_jobs = torch.cuda.device_count() * 2  # 每个 GPU 运行 2 个任务
    
    def save_study(study, trial):
        joblib.dump(study, "optuna_results/study_checkpoint.pkl")

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[save_study])

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # 使用最佳超参数训练最终模型
    best_config = GUTNetConfig(
        input_dim=11,
        num_classes=5,
        hidden_size=trial.params['hidden_size'],
        num_hidden_layers=trial.params['num_hidden_layers'],
        num_attention_heads=1,
        intermediate_size=trial.params['intermediate_size'],
        hidden_dropout_prob=trial.params['hidden_dropout_prob'],
        attention_probs_dropout_prob=trial.params['attention_probs_dropout_prob'],
        problem_type="single_label_classification",
        log_level="INFO"
    )

    data_path = 'data/standardized'
    final_trainer = GUTNetTrainer(best_config, data_path, batch_size=trial.params['batch_size'], lr=trial.params['lr'])
    final_trainer.train(num_epochs=50, patience=10)

    # 在测试集上评估
    test_accuracy = final_trainer.evaluate(final_trainer.test_loader)
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    # 设置CUDA设备可见性
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # 添加这行来检查可用的 GPU 数量
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    
    # 确保存储目录存在
    os.makedirs('optuna_results', exist_ok=True)
    
    # 使用多进程来充分利用 GPU
    num_processes = torch.cuda.device_count() * 2
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(run_optimization, [100 // num_processes] * num_processes)
    pool.close()
    pool.join()