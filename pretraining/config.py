import copy
import argparse

class Configuration:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        
        # config parsed by the default parser
        self._config = None

        # individual configurations for different runs
        self._configs = []
        
        # arguments with more than one value
        self._multivalue_args = []       
        
    def parse(self):
        self._config = self._parser.parse_args()
    
        # find values with more than one entry
        dict_config = vars(self._config)
        for k in dict_config :
            if isinstance(dict_config[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._config)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_config[ma]:
                    connectionrent = copy.deepcopy(c)
                    setattr(connectionrent, ma, v)
                    new_configs.append(connectionrent)

            # store splitted values
            self._configs = new_configs
        
    def get_configs(self):
        return self._configs


def setup_config(config):
    print('Configuration setup ...')
    config._parser.add_argument("--max_epochs", type=int, default=1, nargs='*')
    config._parser.add_argument("--patience", type=int, default=10)
    config._parser.add_argument("--lr", type=float, default=1e-3, nargs='*')
    config._parser.add_argument("--dropout", type=float, default=0.1, nargs='*')
    config._parser.add_argument("--weight_decay", type=float, default=1e-4, nargs='*')
    config._parser.add_argument("--embed_dim", type=int, default=64, nargs='*')
    config._parser.add_argument("--batch_size", type=int, default=64*4, nargs='*')
    config._parser.add_argument("--seed", type=int, default=42, nargs='*')
    config._parser.add_argument("--resultTable", type=str, default="ir2vec_pretrain")
    config._parser.add_argument("--model_name", type=str, default="IR2Vec", help="Name of the model") # InterRoute2Vec
    config._parser.add_argument("--device", type=str, default="cuda:0")
    config._parser.add_argument("--num_logs_batch", type=int, default=50, help="Number of batches to log during training")
    config._parser.add_argument("--expid", type=int, default=-1)
    config._parser.add_argument("--debug", action="store_true", help="Whether to enable debug mode")
    config._parser.add_argument("--recompute_loader", action="store_true", help="Whether to recompute data loaders")
    config._parser.add_argument('--num_samples_debug', type=int, default=10, help='Number of samples to use in debug mode', nargs='*')
    config._parser.add_argument("--max_route_length", type=int, default=128, help="Maximum number of route steps kept per sample", nargs='*')
    config._parser.add_argument("--num_modes", type=int, default=6, help="Number of transport modes encoded in mode vectors", nargs='*')

    config._parser.add_argument("--data_splits_dir", type=str, default="data/splits", help="Directory containing train.pkl, val.pkl, and test.pkl")
    config._parser.add_argument("--dataset_cache_dir", type=str, default="data/cache", help="Directory used to cache built dataset objects")
    config._parser.add_argument("--graph_embeddings_path", type=str, default="data/embeddings/graph_embeddings.pkl", help="Pickle with graph node and edge embeddings")
    config._parser.add_argument("--cell_embeddings_path", type=str, default="data/embeddings/cell_embeddings.pkl", help="Pickle with grid or cell embeddings")
    config._parser.add_argument("--node2cell_path", type=str, default="data/embeddings/node2cell.pkl", help="Pickle with node-to-cell mapping")
    config._parser.add_argument("--edge_features_path", type=str, default="data/features/edge_features.pkl", help="Pickle with edge feature table")
    config._parser.add_argument("--node_features_path", type=str, default="data/features/node_features.pkl", help="Pickle with node feature table")
    config._parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretraining", help="Directory for pretraining checkpoints and arguments")
    config._parser.add_argument("--best_model_path", type=str, default=None, help="Optional explicit path for the best pretrained model checkpoint")

    config._parser.add_argument("--view1_strategies", type=str, default="['tc', 'cm']", help="List of strategies for view 1, e.g., ['tc', 'cm']", nargs='*')   # rm: random, tc: truncate, cm: consecutive
    config._parser.add_argument("--view2_strategies", type=str, default="['rm', 'tc', 'cm']", help="List of strategies for view 2, e.g., ['rm', 'tc', 'cm']", nargs='*')   # rm: random, tc: truncate, cm: consecutive

    config._parser.add_argument("--view1_ratios", type=str, default="[0.3, 0.3]", help="List of strategies for view 1, e.g., ['tc', 'cm']", nargs='*')   # rm: random, tc: truncate, cm: consecutive
    config._parser.add_argument("--view2_ratios", type=str, default="[0.3, 0.3, 0.3]", help="List of strategies for view 2, e.g., ['rm', 'tc', 'cm']", nargs='*')   # rm: random, tc: truncate, cm: consecutive
    config._parser.add_argument('--num_hyper_layers', type=int, default=2, help='Number of hypergraph layers', nargs='*')
    config._parser.add_argument('--num_heads', type=int, default=1, help='Number of heads', nargs='*')

    config._parser.add_argument("--exp_comment", type=str, default="", help="Comment to describe the experiment")
    config._parser.add_argument("--num_log_steps", type=int, default=5, help="Number of steps to log training progress")
    config._parser.add_argument("--losses", type=str, default="['node', 'edge', 'route', 'cons', 'sim', 'clust']", help="Comment to describe the experiment", nargs='+')
    
    config._parser.add_argument("--eval_metrics_weights_retrieval", type=str, default="{'recall@k': 0.4, 'precision@k': 0.4, 'trustworthiness': 0.2}", help="Weights for evaluation metrics in retrieval task", nargs='*')
    config._parser.add_argument("--eval_metrics_weights_ranking", type=str, default="{'spearman': 0.5, 'kendall': 0.5}", help="Weights for evaluation metrics in ranking task", nargs='*')
    config._parser.add_argument("--eval_metrics_weights_clustering", type=str, default="{'silhouette': 0.4, 'ARI': 0.3, 'Gap': 0.3}", help="Weights for evaluation metrics in clustering task", nargs='*')
    config._parser.add_argument("--eval_metrics_weights_global", type=str, default="{'retrieval': 0.7, 'ranking': 0.1, 'clustering': 0.2}", help="Weights for evaluation metrics in global task", nargs='*') # block ranking to 0.1, default 0.7, 0.2 for retrieval and clustering
    config._parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss: 0.3 - 0.7, lower --> loss closer to 1.0", nargs='*')


    config.parse()
    
