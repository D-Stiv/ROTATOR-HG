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
    print('[Evaluation] Configuration setup ...')

    config._parser.add_argument("--exp_comment", type=str, default="", help="Comment to describe the experiment")
    config._parser.add_argument("--eval_type", type=str, default="unsupervised", help="evaluation types: unsupervised, semi-supervised, supervised", nargs='*')
    config._parser.add_argument("--model_name", type=str, default="ir2vec", help="model names to evaluate", nargs='*')
    config._parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for evaluation (e.g., 'cuda:0' or 'cpu')")
    config._parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility", nargs='*')
    config._parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation", nargs='*')
    config._parser.add_argument("--debug", action="store_true")

    # finetune arguments for supervised evaluation
    config._parser.add_argument("--finetune_epochs", type=int, default=50, help="Epoch count for supervised fine-tuning when a cached head is not reused", nargs='*')
    config._parser.add_argument("--finetune_patience", type=int, default=50, help="Patience for supervised fine-tuning when a cached head is not reused", nargs='*')
    config._parser.add_argument("--finetune_batch_size", type=int, default=2048, help="Batch size for supervised fine-tuning", nargs='*')
    config._parser.add_argument("--finetune_hidden_dim", type=int, default=128, help="Hidden dimension for supervised fine-tuning head", nargs='*')
    config._parser.add_argument("--finetune_dropout", type=float, default=0.1, help="Dropout rate for supervised fine-tuning head", nargs='*')
    config._parser.add_argument("--finetune_lr", type=float, default=1e-3, help="Learning rate for supervised fine-tuning when a cached head is not reused", nargs='*')
    config._parser.add_argument("--finetune_weight_decay", type=float, default=1e-5, help="Weight decay for supervised fine-tuning when a cached head is not reused", nargs='*')
    config._parser.add_argument("--finetune_regression_loss", type=str, default="mae", help="Regression loss function for supervised fine-tuning", nargs='*')

    config._parser.add_argument("--supervised_target", type=str, default="accident_score", help="Target variable for supervised evaluation: accident_score, accident_label, criteria_scores, criteria_labels", nargs='*')
    config._parser.add_argument("--supervised_task_names", type=str, default=None, help="List of tasks for supervised evaluation", nargs='*')


    config.parse()
    
