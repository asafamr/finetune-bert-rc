import asyncio
import pipes
import json
import fnmatch
from contextlib import closing


def isnotebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class ExperimentSubmission:
    def __init__(self, image: str, result_path: str, arguments: list, sources: dict = None, desc=None,
                 env_vars: dict = None, name: str = None,
                 group: str = None, gpu_count: int = None, gpu_type: str = None, cpu: float = None, memory: str = None):
        """

        :param image: image id/name: bp_asdqawedrqf / lme-testv1
        :param result_path: '/output'
        :param sources: e.g. {'ds_abcd1234':'/data','my_dataset':'/data2'}
        :param arguments: e.g. ['--some-arg',123]
        :param desc: str
        :param env_vars: e.g. {'SOME_VAR':1234} <- these show up in the UI
        :param name: 'None' to assign automatically
        :param group: set experiment group after creation
        :param gpu_count: GPUs to use for this experiment (e.g., 2)
        :param gpu_type: GPU type to use for this experiment (e.g., 'p100' or 'v100')
        :param cpu: CPUs to reserve for this experiment (e.g., 0.5)
        :param memory: Memory to reserve for this experiment (e.g., 1GB)

        """
        assert image
        assert result_path
        assert sources
        self.desc = desc
        self.image = image
        self.result_path = result_path
        self.sources = sources
        self.arguments = arguments
        self.env_vars = env_vars
        self.name = name
        self.group = group
        self.gpu_count = gpu_count
        self.gpu_type = gpu_type
        self.cpu = cpu
        self.memory = memory


class BeakerAPI:
    def __init__(self, n_concurrent_request=5, beaker_path='beaker'):
        self.concurrency_lim_sem = asyncio.BoundedSemaphore(n_concurrent_request)
        self.beaker_path = beaker_path

    async def assert_group(self, group_name, desc=''):
        try:
            await self._run_beaker_command('group inspect '.split() + [group_name], False)
        except Exception as e:
            if 'could not resolve group reference' in str(e):
                await self._run_beaker_command('group create --name'.split() + [group_name, '--desc', desc], False)
                return True
        return False

    async def create_experiment(self, exp: ExperimentSubmission):
        args = 'experiment run --quiet'.split()
        group_name = None
        program_args = []
        for prop, val in vars(exp).items():
            if val is None:
                continue
            if prop == 'sources':
                for sourceid, mount in val.items():
                    args += ['--source', f'{sourceid}:{mount}']
                continue
            if prop == 'env_vars':
                for envvarkey, envvarval in val.items():
                    args += ['--env', f'{envvarkey}={envvarval}']
                continue
            if prop == 'arguments':
                program_args = val
                continue
            if prop == 'group':
                group_name = val
                continue
            prop = prop.replace('_', '-')
            args += ['--' + prop, str(val)]

        exper_id = await self._run_beaker_command(args + program_args, False)
        if group_name:
            await self._run_beaker_command(['group', 'add', group_name, exper_id], False)
        return exper_id

    async def report_group(self, group_name, globs=None):
        assert globs is None or globs is isinstance(globs, list)
        contents = await self._run_beaker_command('group inspect --contents '.split() + [group_name])
        exper_ids = contents[0]['experiments']
        print(f'retrieving {len(exper_ids)} experiments')
        expers = []

        async def get_batch(batch_ids):
            batch = await self._run_beaker_command('experiment inspect '.split() + batch_ids)
            print('.', end='')
            return batch

        for start in range(0, len(exper_ids), 5):
            expers.append(get_batch(exper_ids[start:start + 5]))
        expers = await asyncio.gather(*expers)
        expers = [y for x in expers for y in x]  # flatten

        print('\nretrieving tasks...')
        tasks_ids = [y['taskId'] for x in expers for y in x['nodes']]
        results_ids = [y['resultId'] for x in expers for y in x['nodes']]
        tasks = self._run_beaker_command('task inspect '.split() + tasks_ids)

        if globs:
            results_files = asyncio.gather(
                *[self._run_beaker_command('dataset ls '.split() + [x], False) for x in results_ids])
            results_files, tasks = await  asyncio.gather(results_files, tasks)
            files_to_download = []
            for rows, rid in zip(results_files, results_ids):
                rows = rows.split('\n')[:-1]
                for row in rows:
                    size, ctime, filename = row.strip().split(maxsplit=2)
                    for glob in globs:
                        if fnmatch.fnmatch(filename, glob):
                            if BeakerAPI._is_big_file_size(size):
                                print(f'filename {filename} is too big: {size}, skipping...')
                                continue
                            files_to_download.append((rid, filename))

            downloaded_files = await asyncio.gather(
                *[self._run_beaker_command('dataset stream-file '.split() + [resultId, filename],
                                           isjson=('.json' in filename.lower())) for
                  resultId, filename in files_to_download])
        else:
            tasks = await tasks
            files_to_download = downloaded_files = []

        final_results = {}
        for exper in expers:
            exper['files'] = {}
            exper_results_id = [x['resultId'] for x in exper['nodes']]
            for file_content, (resultId, filename) in zip(downloaded_files, files_to_download):
                if resultId in exper_results_id:
                    exper['files'][filename] = file_content
            final_results[exper['id']] = exper
        for task in tasks:
            final_results[task['experimentId']]['task'] = task

        return final_results

    @staticmethod
    def _is_big_file_size(filesize):
        return any(x in filesize for x in ['MB', 'GB', 'TB', 'PB', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'])

    async def _run_beaker_command(self, args, isjson=True):
        command = self.beaker_path + ' ' + ' '.join(pipes.quote(str(x)) for x in args)
        async with self.concurrency_lim_sem:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            stderr = stderr.decode()
            stdout = stdout.decode()
            if stderr:
                raise Exception(stderr)
            if isjson:
                return json.loads(stdout)
            return stdout

    def do_stuff_sync(self, async_callback):
        if isnotebook():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(async_callback(self))
        else:
            with closing(asyncio.get_event_loop()) as loop:
                return loop.run_until_complete(async_callback(self))


from itertools import product


def product_generator(**params):
    filed_names, arrs = list(zip(*params.items()))
    for test in product(*arrs):
        params = dict(zip(filed_names, test))
        yield params


async def my_grid_search(beaker):
    group_name = 'tacred_bool_lme_vs_ft_v2'

    await beaker.assert_group(group_name, 'TACRED bool RC: LME vs finetuned LM')

    def create_exper(method, seed, rel, num_pos):
        num_neg = num_pos * 10
        if num_neg > 200:
            num_neg = 200
        new_exper = ExperimentSubmission(image='lme-v1.01', result_path='/output', desc='part of ' + group_name,
                                         sources={'TACRED': '/data'},
                                         env_vars={'rel': rel, 'seed': seed, 'method': method,
                                                   'num_pos': num_pos, 'num_neg': num_neg},
                                         gpu_count=1,
                                         gpu_type='v100', group=group_name, arguments=[])

        args = ['python']
        if method.startswith('lme'):
            args += (f'''transformers_rc_lme.py tacred --model_name=xlnet --nopbar=True --score_bool_probs=True
                      --output_dir=/output  --seed={seed} --data_dir=/data
                      --task_target_rel={rel} --task_seed={seed}  --batch_max_num_tokens=6000
                      --task_num_samps_pos_train={num_pos} --task_num_samps_neg_train={num_neg}
                      '''.split())

            if method == 'lme-5':
                args += ['--topk_features', 5]
            elif method == 'lme-10-B':
                args += ['--topk_features', 10]
        else:
            epochs = 200
            if num_pos <= 20:
                epochs = 500
            args += (f'''transformers_rc_finetune.py --data_dir=/data --output_dir=/output --model_name=xlnet --fp16=True  
                      --max_seq_length=220 --do_train=True --eval_test=True --per_gpu_batch_size=8
                      --overwrite_output_dir=True --task=tacredbool --num_train_epochs={epochs} --save_steps=-1
                      --logging_steps=100 --evaluate_during_training=True --stop_train_low_score_k=5
                      --task_target_rel={rel} --task_seed={seed} 
                      --task_num_samps_pos_train={num_pos} --task_num_samps_neg_train={num_neg}
                      --seed={seed} --nopbar=True --score_bool_probs=True
                      '''.split())

            # task_seed={seed}
        new_exper.arguments = args
        return new_exper

    all_expers = []
    for exper_settings in product_generator(seed=[102],  # 101],
                                            method=['lme-10-B', 'finetune-B'],
                                            # rel=['org:top_members/employees', 'org:country_of_headquarters',
                                            # 'per:title'],
                                            rel=['org:alternate_names',  # 'org:city_of_headquarters', 'org:members',
                                                 # 'org:parents',
                                                 # 'org:subsidiaries',
                                                 'per:age', 'org:top_members/employees',
                                                 'org:country_of_headquarters', 'per:title',
                                                 # 'per:cause_of_death',
                                                 'per:cities_of_residence',
                                                 'per:countries_of_residence',  # 'per:date_of_death',
                                                 'per:employee_of',
                                                 'per:origin',
                                                 # 'per:other_family',
                                                 # 'per:spouse',
                                                 'per:stateorprovinces_of_residence'
                                                 ],
                                            num_pos=[5, 10, 30, 50, 100, 300]  # , 10, 50],
                                            ):
        all_expers.append(beaker.create_experiment(create_exper(exper_settings)))

    return await asyncio.gather(*all_expers)


if __name__ == '__main__':
    beaker = BeakerAPI(beaker_path='beaker')
    # print(beaker.do_stuff_sync(add_submissions))
    beaker.do_stuff_sync(my_grid_search)
