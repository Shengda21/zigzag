"""Optional shared-training/exclusive-inference lock for two independent cases."""
import contextlib,os


@contextlib.contextmanager
def gpu_access(script):
    path=os.environ.get('SIMODS_GPU_GATE')
    training=script in ('train_components.py','train_exchange.py');evaluation=script in ('evaluate_components.py','evaluate_exchange.py','rollout.py')
    if not path or not(training or evaluation):yield;return
    import fcntl
    with open(path,'a+') as gate:
        fcntl.flock(gate,fcntl.LOCK_SH if training else fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(gate,fcntl.LOCK_UN)
