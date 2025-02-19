import os
import csv
import numpy
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



def read_csv(exp_id, runs):
    datas = {}
    for run in runs:
        if run[:8] == 'gaussian':
            continue
        if run[:6] == '.hydra':
            continue
        if run[:11] == 'checkpoints':
            continue
        datas[run] = {}
        base_dir = f'runs/{exp_id}/{run}/scalars'
        subdirs = os.listdir(base_dir)
        for subdir in subdirs:
            csv_files = os.listdir(base_dir + '/' + subdir)
            for csv_name in csv_files:
                with open(f'{base_dir}/{subdir}/{csv_name}') as f:
                    reader = csv.reader(f)
                    data = []
                    for row in reader:
                        data.append(row)
                    data = [[float(e) for e in row] for row in data[1:]]
                    datas[run][subdir + '/' + csv_name] = data
    return datas
    
def group_datas(datas):
    runs = datas.keys()
    groups = {}
    for run in runs:
        seed_str_len = len(run.split('_')[-1])
        config = run[:-seed_str_len-1]
        if config not in groups:
            groups[config] = []
        groups[config].append(run)
    # grouped_datas = {}
    # for group in groups.keys():
    #     grouped_datas[group] = []
    #     for run in groups[group]:
    #         grouped_datas[group].extend(datas[run]['charts/episodic_return'])
    # return grouped_datas

    return groups

def take_mean_stds(groups, datas, key="charts/train_return"):
    means_by_group = {}
    stds_by_group = {}

    for config in groups.keys():
        print('\t', config, key.split('/')[-1])
        if config == 'gaussian':
            values_det = []
        values = []

        for run in groups[config]:
            # key = input(f'key for run {run}: ')
            tuples = datas[run][key]
            values.append([])
                
            for tup in tuples:
                values[-1].append(tup[-1])

            # if config == 'gaussian':
            #     tuples_det = datas[run][key]
            #     values_det.append([])

            # for tup in tuples_det:
            #     values_det[-1].append(tup[2])

        print('\t\t', [len(val) for val in values])
        max_len = max([len(val) for val in values])
        max_len = 100
        if any([not len(val) == max_len for val in values]):
            print('WARNING: INHOMOGINEOUS SHAPE OF RESULTS')
        means = []
        stds = []
        for i in range(max_len):
            datas = []
            for val in values:
                try:
                    datas.append(val[i])
                except:
                    pass
            if len(datas) > 0:
                means.append(numpy.mean(datas, axis=0))
                stds.append(numpy.std(datas, axis=0))
            # else:
            #     means.append(means[-1])
            #     stds.append(stds[-1])


        # means = numpy.mean(values, axis=0)
        # stds = numpy.std(values, axis=0)

        means_by_group['gaussian'] = numpy.array(means)
        stds_by_group['gaussian'] = numpy.array(stds)
        # means_by_group[config] = means
        # stds_by_group[config] = stds

    return means_by_group, stds_by_group

def plot_results(means, stds, base_dir):
    for config in means.keys():
        mean_datas = means[config]
        std_datas = stds[config]

        x = numpy.linspace(0, 1, len(mean_datas))

        # plt.ylim((100, -1600))
        plt.plot(x, mean_datas)
        # plt.legend()
        os.makedirs(f'{base_dir}/{config}', exist_ok=True)
        plt.savefig(f'{base_dir}/{config}/means.pdf')
        plt.savefig(f'{base_dir}/{config}/means.png')

        plt.clf()

def plot_results_grouped(means, stds, base_dir):
    configs = means.keys()
    baseline = None
    for config in configs:
        mean_datas = means[config]
        std_datas = stds[config]

        x = numpy.linspace(0, 1, len(mean_datas))

        plt.plot(x, mean_datas, label=config)
        plt.fill_between(x, mean_datas + std_datas, mean_datas - std_datas, alpha=0.1)
    plt.legend()
    plt.xlabel("Interaction (steps, )")
    plt.ylabel("Episode Return")
    plt.title("Episode Return during TRAINING")
    # plt.yscale("symlog")
    # plt.ylim(-1000, 10)
    # os.makedirs(f'{base_dir}/{fc_config}', exist_ok=True)
    os.makedirs(f'{base_dir}', exist_ok=True)
    plt.savefig(f'{base_dir}/evals.pdf')
    # plt.savefig(f'{base_dir}/means.png')


def tfboard2csv(event_acc, path):
    tags = event_acc.Tags()['scalars']
    for tag in tags:
        filepath = f'runs/{path}/scalars/{tag}'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['wall_time', 'step', 'value'])
            events = event_acc.Scalars(tag)
            # if path[-3:] == '0_3':
            #     print(path)
            # if path=='20240111-185203/fc32x2_nfx0_3':
            #     print(events)
            for event in events:
                csv_writer.writerow([event.wall_time, event.step, event.value])


# os.chdir('..')
# exp_ids = ['thesis/BipedalWalker-v3-OURS-pibe-0.2-sb3hyperparams_nf', 'thesis/BipedalWalker-v3-SAC-sb3hyperparams_nf', 'thesis/BipedalWalker-v3-PPO-sb3hyperparams_nf']
# exp_ids = ['thesis/BipedalWalker-v3-OURS-pibe-0.0-sb3hyperparams_nf', 'thesis/BipedalWalker-v3-OURS-pibe-0.2-sb3hyperparams_nf', 'thesis/BipedalWalker-v3-OURS-pibe-0.4-sb3hyperparams_nf']
# exp_ids = ['thesis/LunarLander-v2-OURS-sb3hyperparams_nf', 'thesis/LunarLander-v2-SAC-sb3hyperparams_nf']
exp_ids = ['AbsEnv-v0-step10-clip-long-OURS-0.001_gauss', 'ablation-AbsEnv-v0-OURS-0.001_nf']
# exp_ids = ['AbsEnv-v0-extralong-SAC-0.01_gauss', 'AbsEnv-v0-extralong-A2C-0.01_gauss',  'AbsEnv-v0-extralong-OURS-0.01_gauss']
# exp_ids = ['AbsExploreEnv-v0-OURS-0.001_gauss', 'AbsExploreEnv-v0-OURS-0.001_nf', 'AbsExploreEnv-v0-OURS-0.001_pibe0.0_gauss', 'AbsExploreEnv-v0-OURS-0.001_pibe0.0_nf']
# exp_ids = ['AbsExploreEnv-v0-OURS-0.001_gauss', 'AbsExploreEnv-v0-OURS-0.001_nf']

plot_datas = {}

for exp_id in exp_ids:
    runs = os.listdir(f'runs/{exp_id}')
    print(f'  {len(runs)} experiments detected.')

    skip_exp = False

    for run in runs:
        # if run[-3:] == '0_3':
        files = [f for f in os.listdir(f'runs/{exp_id}/{run}') if os.path.isfile(os.path.join(f'runs/{exp_id}/{run}', f))]
        files = [f for f in files if not f[-3:] == 'png']
        files = [f for f in files if not f[-3:] == 'dra']
        try:
            event_acc = EventAccumulator(f'runs/{exp_id}/{run}/{files[-1]}')
            event_acc.Reload()
            tfboard2csv(event_acc, exp_id+'/'+run)
        except:
            print(f'Skipping {exp_id}/{run}')

    datas = read_csv(exp_id, runs)
    groups = group_datas(datas)

    # means_b, stds_b = take_mean_stds(groups, datas, 'charts/pi_b_return')
    means_e, stds_e = take_mean_stds(groups, datas, 'eval/ep_r')

    means = {
        # 'pi_b': means_b['gaussian'],
        'pi_eval': means_e['gaussian']
    }

    stds = {
        # 'pi_b': stds_b['gaussian'],
        'pi_eval': stds_e['gaussian']
    }

    plot_datas[exp_id] = {
        'means': means,
        'stds': stds
    }

    base_dir = f'results/{exp_id}'



# for exp_id in exp_ids:
#     # plt.plot(x, plot_datas[exp_id]['means']['pi_b'], label=exp_id + '_pi_b')
#     print(exp_id)
#     x = numpy.linspace(0, 500000, 100)
#     if not exp_id == 'thesis/BipedalWalker-v3-PPO-sb3hyperparams_nf':
#         x = x[:len(plot_datas[exp_id]['means']['pi_eval'])]
#     else:
#         x = numpy.linspace(0, 500000, len(plot_datas[exp_id]['means']['pi_eval']))
#     exp_label = exp_id.split('_')[0].split('-')[-2]
#     # if exp_label == '0.2':
#     #     exp_label = 'OURS'
#     plt.plot(x, plot_datas[exp_id]['means']['pi_eval'], label=exp_label)
#     # plt.fill_between(x, plot_datas[exp_id]['means']['pi_b'] + plot_datas[exp_id]['stds']['pi_b'], plot_datas[exp_id]['means']['pi_b'] - plot_datas[exp_id]['stds']['pi_b'], alpha=0.1)
#     plt.fill_between(x, plot_datas[exp_id]['means']['pi_eval'] + plot_datas[exp_id]['stds']['pi_eval'], plot_datas[exp_id]['means']['pi_eval'] - plot_datas[exp_id]['stds']['pi_eval'], alpha=0.1)

exp_id = 'AbsEnv-v0-step10-clip-long-OURS-0.001_gauss'
x = numpy.linspace(0, 100000, len(plot_datas[exp_id]['means']['pi_eval']))
plt.plot(x, plot_datas[exp_id]['means']['pi_eval'], label='OURS')
plt.fill_between(x, plot_datas[exp_id]['means']['pi_eval'] + plot_datas[exp_id]['stds']['pi_eval'], plot_datas[exp_id]['means']['pi_eval'] - plot_datas[exp_id]['stds']['pi_eval'], alpha=0.1)

exp_id = 'ablation-AbsEnv-v0-OURS-0.001_nf'
x = numpy.linspace(0, 100000, len(plot_datas[exp_id]['means']['pi_eval']))
plt.plot(x, plot_datas[exp_id]['means']['pi_eval'], label='OURS (pi_e = pi_b)')
plt.fill_between(x, plot_datas[exp_id]['means']['pi_eval'] + plot_datas[exp_id]['stds']['pi_eval'], plot_datas[exp_id]['means']['pi_eval'] - plot_datas[exp_id]['stds']['pi_eval'], alpha=0.1)


# for exp_id in ['AbsEnv-v0-step10-clip-long-SAC-0.001_gauss', 'redo-AbsEnv-v0-A2C-0.001_nf', 'AbsEnv-v0-step10-clip-long-PPO-0.001_gauss']:
#     # plt.plot(x, plot_datas[exp_id]['means']['pi_b'], label=exp_id + '_pi_b')
#     print(exp_id)
#     x = numpy.linspace(0, 100000, len(plot_datas[exp_id]['means']['pi_eval']))
#     exp_label = exp_id.split('_')[0].split('-')[-2]
#     plt.plot(x, plot_datas[exp_id]['means']['pi_eval'], label=exp_label, linestyle='dashed')
#     # plt.fill_between(x, plot_datas[exp_id]['means']['pi_b'] + plot_datas[exp_id]['stds']['pi_b'], plot_datas[exp_id]['means']['pi_b'] - plot_datas[exp_id]['stds']['pi_b'], alpha=0.1)
#     plt.fill_between(x, plot_datas[exp_id]['means']['pi_eval'] + plot_datas[exp_id]['stds']['pi_eval'], plot_datas[exp_id]['means']['pi_eval'] - plot_datas[exp_id]['stds']['pi_eval'], alpha=0.1)

plt.legend()
plt.xlabel("Interaction (steps)")
plt.ylabel("Episode Return")
# plt.title("Episode Return")
# plt.yscale("log")
# plt.yscale("symlog")
# plt.ylim(-1, 1)
# os.makedirs(f'{base_dir}/{fc_config}', exist_ok=True)
base_dir = base_dir.split('_')[0]
base_dir = 'AbsEnv'
os.makedirs(f'{base_dir}', exist_ok=True)
# plt.show()
plt.savefig(f'{base_dir}/evals.pdf')