"""Wall-clock allocation across folds. This never launches training itself."""
import json
import time
from pathlib import Path


class RunBudget:
    def __init__(self, path, limit_seconds, resume=False, clock=None):
        self.path = Path(path)
        self.limit = float(limit_seconds)
        if self.limit <= 0:
            raise ValueError('Time budget must be positive.')
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.previous = 0.
        if self.path.exists():
            if not resume:
                raise FileExistsError(f'Budget state exists at {self.path}; resume or use another run directory.')
            state = json.loads(self.path.read_text(encoding='utf-8'))
            if state['limit_seconds'] != self.limit:
                raise ValueError('Resume time budget differs from the saved run.')
            self.previous = state['elapsed_seconds']

    @property
    def elapsed(self):
        return self.previous + self.clock()-self.started

    @property
    def remaining(self):
        return max(0.,self.limit-self.elapsed)

    def stage_deadline(self, weights, reserve_seconds=60):
        if not weights or min(weights)<=0:
            raise ValueError('Stage weights must be positive.')
        allowance = self.remaining * weights[0]/sum(weights)
        if allowance <= reserve_seconds+30:
            raise TimeoutError('Too little time left to train and validate all remaining stages.')
        return self.clock()+allowance-reserve_seconds

    def save(self, **extra):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        state = dict(limit_seconds=self.limit,elapsed_seconds=self.elapsed,remaining_seconds=self.remaining,**extra)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(self.path)


def planned_steps(manifest,config,folds,stages=('coarse','fine')):
    import math
    result=[]
    for fold in folds:
        count=sum(r['split']=='train' and r['fold']!=fold for r in manifest['records'])
        for stage in stages:
            settings=config[stage]
            per_epoch=math.ceil(count*settings.get('repeats',1)/config['batch_size'])
            result.append(dict(fold=fold,stage=stage,train_images=count,steps_per_epoch=per_epoch,
                               max_epochs=settings['epochs'],max_steps=per_epoch*settings['epochs']))
    return dict(stages=result,max_optimizer_steps=sum(x['max_steps'] for x in result),
                time_budget_seconds=config.get('time_budget_seconds'),
                note='Step counts are exact upper bounds for these settings; wall time and accuracy are unmeasured.')


def fit_epoch_limit(configured,completed,seconds_per_epoch,remaining_seconds):
    if seconds_per_epoch<=0:
        raise ValueError('Measured epoch duration must be positive.')
    # Five percent throughput margin, and never lengthen the configured schedule.
    affordable=int(max(0.,remaining_seconds)/(seconds_per_epoch*1.05))
    return min(configured,completed+affordable)
