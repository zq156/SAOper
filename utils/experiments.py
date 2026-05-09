import numpy as np

def topk_state_quality(observations, actions, reward_func, 
                       is_task_relevant_fn, topk_ratio=0.1):
    """
    计算 Top-K 状态质量（Precision@K）

    Args:
        observations: [N, ...]
        actions: [N, ...]
        reward_func: 你的 reward model
        is_task_relevant_fn: 判定函数
        topk_ratio: 例如 0.1 表示 top 10%

    Returns:
        precision_at_k
    """

    N = len(observations)
    rewards = []

    # 1. 计算所有 reward
    for i in range(N):
        r = reward_func.step(observations[i], actions[i])
        rewards.append(r)

    rewards = np.array(rewards)

    # 2. 取 Top-K index
    K = int(N * topk_ratio)
    topk_indices = np.argsort(rewards)[-K:]

    # 3. 统计 task-relevant
    relevant_count = 0
    for idx in topk_indices:
        if is_task_relevant_fn(observations[idx], actions[idx]):
            relevant_count += 1

    precision_at_k = relevant_count / K

    return precision_at_k