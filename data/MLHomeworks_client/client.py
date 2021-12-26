# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import requests
import glob
import sys
import os
from tqdm import tqdm

from group_info import SID, TOKEN

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f :
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def main(ip="115.236.52.125", port="4000", sid=SID, token=TOKEN,
         ans=None, problem="Toxicity_evaluate", verbose=1):
    if verbose: print("正在提交...")
    url = "http://%s:%s/jsonrpc" % (ip, port)
    
    payload = {
        "method": problem,
        "params": [ans],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url,
        json=payload,
        headers={"token": token, "sid": sid}
    ).json()
    
    
    if verbose: print(response)
    if "auth_error" in response:
        print("您的认证信息有误")
        return response["auth_error"]
    elif "error" not in response:
        if verbose: print("测试完成，请查看分数")
        return response["result"]
    else:
        print("提交文件存在问题，请查看error信息")
        return response["error"]["data"]["message"]

def submission_path_to_df(path):
    return pd.DataFrame(pd.read_csv(path, header=None, sep=" "))

def df_to_lines(df):
    return [f"{int(sample_id)} {sample_pred}" for sample_id, sample_pred in [
        df.iloc[i].values.tolist() for i in range(df.shape[0])
    ]]

def ensemble_submissions(submission_paths, weights="mean", dfs=None):
    if dfs is None:
        return_df = False
        dfs = [submission_path_to_df(path) for path in submission_paths]
        # 各个模型得到的输出
    else:
        return_df = True
    
    df_ensemble = pd.DataFrame()
    df_ensemble[0] = dfs[0][0]
    if weights=="mean" or sum(weights)==0:
        df_ensemble[1] = sum([df[1] for df in dfs]) / len(dfs)
        # 取平均进行集成
    elif type(weights)==list:
        norm = sum(weights)
        weights = [_/norm for _ in weights]
        df_ensemble[1] = sum([dfs[i][1]*weights[i] for i in range(len(dfs))])
    
    if return_df:
        return df_to_lines(df_ensemble), df_ensemble
    else:
        return df_to_lines(df_ensemble)

class GeneticAlgorithm4Submission:
    def __init__(self, max_iter, num_caches, submission_paths):
        self.max_iter = max_iter
        self.num_caches = num_caches
        self.submission_paths = submission_paths
        self.best_score = 0
        self.init_test()
    
    def run(self):
        self.update()
        for i in tqdm(range(int(self.max_iter/2), self.max_iter)):
            seed_decay = (np.cos(np.pi/self.max_iter*i)+1) * 0.2
            p_decay = np.cos(np.pi/self.max_iter*i)+1
            self.iter_step(seed_decay=seed_decay, p_decay=p_decay)
            if i%10==0:
                print("\n".join([f"{score:.8f}" for df,score in self.caches]))
        best_lines = df_to_lines(self.caches[0][0])
        main(ans=best_lines)
        with open(f"./{self.best_score:.8f}_submission.txt", "w") as f:
            f.write("\n".join([line for line in best_lines]))
        self.explain(self.caches[0][0])
    
    def init_test(self):
        path_to_df = {path:submission_path_to_df(path) for path in submission_paths}
        path_scores = [(path,main(ans=df_to_lines(path_to_df[path]), verbose=0)) for path in tqdm(path_to_df)]
        path_scores = sorted(path_scores, key=lambda x:x[1], reverse=True)
        print("initial scores:")
        for path,score in path_scores:
            print(f"{score:.8f} - {path}")
        self.caches = [(path_to_df[path], score) for path,score in path_scores if score > 0.9]
        self.seeds = self.refine(self.caches)
        self.best_score = self.caches[0][1]
    
    def iter_step(self, seed_decay=1.0, p_decay=1.0):
        for _ in range(5):
            p = 1/(np.log(np.arange(len(self.caches))*p_decay+1)+1)
            index_df_from_cache = np.random.choice(len(self.caches), 3, p=p/p.sum()).tolist()
            df_from_cache = [self.caches[i][0] for i in index_df_from_cache]
            weights_cache = [np.random.normal(1,1) for i in range(len(df_from_cache))]
            
            p = 1/(np.log(np.arange(len(self.seeds))+1)+1)
            index_df_from_seed = np.random.choice(len(self.seeds), 3, p=p/p.sum()).tolist()
            df_from_seed = [self.seeds[i][0] for i in index_df_from_seed]
            weights_seed = [np.random.normal(0,1)*seed_decay for i in range(len(df_from_seed))]
            
            lines, df = ensemble_submissions(_, weights=weights_cache+weights_seed, dfs=df_from_cache+df_from_seed)
            score = main(ans=lines, verbose=0)
            if score > self.best_score:
                print(f"update best score: {self.best_score:.8f} -> {score:.8f}")
                print(f"through: No.{index_df_from_cache+index_df_from_seed} * {weights_cache+weights_seed}")
                self.best_score = score
            self.caches.append((df, score))
        self.update()
    
    def explain(self, target_df):
        a = np.array([df[1].tolist() for df,score in self.seeds]).T
        b = np.array(target_df[1].tolist())
        weights = np.linalg.solve(a[:len(self.seeds),:],b[:len(self.seeds)])
        print(f"solution:")
        for i,(df,score) in enumerate(self.seeds):
            print(f"{score:.8f} * {weights[i]:.3f}")
    
    def refine(self, queue):
        """ 重排序，队列头部仅保留线性无关组 """
        full_rank = []
        to_delete = []
        for i,(df,score) in enumerate(queue):
            pred = df[1].to_list()
            if np.linalg.matrix_rank(np.mat(full_rank+[pred])) == len(full_rank)+1:
                full_rank += [pred]
            else:
                to_delete += [i]
        if len(to_delete)>0:
            print(f"to_delete: {to_delete}")
        to_keep = [i for i in range(len(queue)) if i not in to_delete]
        return [queue[i] for i in to_keep + to_delete]
    
    def update(self):
        self.caches = sorted(self.caches, key=lambda x:x[1], reverse=True)
        self.caches = self.refine(self.caches)
        self.caches = self.caches[:self.num_caches]
    
if __name__ == "__main__":
    # 需要修改的参数：problem, sid, token

    # problem 参数：
    #    Action_evaluate:        低分辨率视频行为识别
    #    FoodPredict_evaluate:   菜品分类
    #    StoreSale_evaluate:     商品销售额预测
    #    Toxicity_evaluate:      恶意评论分类
    #    CarDemand_evaluate:     汽车需求量预测
    #    FineGrainedCar_evaluate:细粒度汽车分类
    #    Traffic_evaluate:       疫情人流量预测
    #    Mask_evaluate:          口罩检测

    problem = "Toxicity_evaluate"
    # IP 固定为 115.236.52.125
    ip = "115.236.52.125"
    # 端口不需要修改
    port = "4000"
    # 改成你的学号
    sid = SID
    # 改成你的口令
    token = TOKEN

    if problem in ["Action_evaluate",
                   'FoodPredict_evaluate',
                   'StoreSale_evaluate',
                   'Toxicity_evaluate',
                   'CarDemand_evaluate',
                   'FineGrainedCar_evaluate',
                   'Traffic_evaluate']:
        
        def get_search_grid(depth, grid, rang=3, base=3):
            if depth==0:
                yield [base**i for i in grid] + [0]
            else:
                for i in range(rang):
                    for _ in get_search_grid(depth-1, grid+[i], rang):
                        yield _
        def search_path(path):
            for subpath in [os.path.join(path, _) for _ in os.listdir(path)]:
                if os.path.isdir(subpath):
                    for _ in search_path(subpath):
                        yield _
                elif "submission.txt" in subpath:
                    yield subpath
                else:
                    pass
        
        submission_path = sys.argv[1]
        if "GA" in submission_path: # ensemble the submissions under specified path through Genetic Algorithm
            submission_paths = [_ for _ in search_path(sys.argv[2])]
            GA = GeneticAlgorithm4Submission(max_iter=int(submission_path[2:]), num_caches=24, submission_paths=submission_paths)
            GA.run()
            
        elif submission_path.isdigit(): # ensemble the specified submissions through grid search
            submission_paths = [sys.argv[i+2] for i in range(int(submission_path))]
            d = ensemble_submissions(submission_paths)
            best_score = 0
            best_weights = None
            from tqdm import tqdm
            for weights in tqdm(get_search_grid(depth=int(submission_path), grid=[]), total=3**int(submission_path)):
                d = ensemble_submissions(submission_paths, weights)
                score = main(ip, port, sid, token, d, problem)
                print(f"{weights}: {score}")
                if score > best_score:
                    best_score = score
                    best_weights = weights
            print(f"best_score: {best_score}")
            print(f"best_weights: {best_weights}")
            print(f"submit this again...")
            main(ip, port, sid, token, ensemble_submissions(submission_paths, best_weights), problem)
            
        elif "+" in submission_path: # ensemble the specified submissions with specified weights
            weights = [int(_) for _ in submission_path.split("+")]
            submission_paths = [sys.argv[i+2] for i in range(int(len(weights)))]
            d = ensemble_submissions(submission_paths, weights)
            score = main(ip, port, sid, token, d, problem)
            print(score)
        
        elif "submission.txt" not in submission_path: # simply post the submissions under specified path
            path_score = []
            for _ in search_path(submission_path):
                import time
                time.sleep(3)
                with open(_) as f:
                    d = list(f.readlines())
                score = main(ip, port, sid, token, d, problem)
                print(_)
                print(score)
                path_score.append((_,score))
            print("\n".join([str(_) for _ in sorted(path_score, key=lambda x:x[1], reverse=True)]))
        
        else: # simply post the single specified submission
            with open(submission_path) as f:
                d = list(f.readlines())
            score = main(ip, port, sid, token, d, problem)
            print(submission_path)
            print(score)

    elif problem == "Mask_evaluate":
        submit_dir = './submission'
        submissions = os.listdir('./submission')
        d = {}
        for submit in submissions:
            submit_file = os.path.join(submit_dir,submit)
            with open(submit_file,'r') as f:
                d[submit] = f.read().splitlines()
        score = main(ip, port, sid, token, d, problem)
        print(score)