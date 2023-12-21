import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmin = result[:, 1].argmin().item()
            print(f'Final results of Run {run + 1:02d}:')
            print(f'Best Epoch: {argmin+1:2d}')
            print(f'Lowest Loss: {result[:, 0].min():.6f}')
            print(f'Lowest Valid MAE: {result[argmin, 1]:.6f}')
            print(f'Final Test MAE: {result[argmin, 2]:.6f}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 1].min().item()
                test = r[r[:, 1].argmin(), 2].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Lowest Valid MAE: {r.mean():.6f} ± {r.std():.6f}')
            r = best_result[:, 1]
            print(f'  Final Test MAE: {r.mean():.6f} ± {r.std():.6f}')
