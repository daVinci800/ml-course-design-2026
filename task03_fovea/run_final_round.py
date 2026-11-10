"""Serial GPU queue with a five-hour round budget and cooperative epoch-boundary pause."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from final_round import JOB, REPORT, ROOT, EXPERIMENTS, read, write


def process_alive(pid):
    if not isinstance(pid,int) or pid <= 0: return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle: return ctypes.get_last_error()==5
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value==259
        finally: kernel.CloseHandle(handle)
    try: os.kill(pid,0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    for stream in (sys.stdout,sys.stderr): stream.reconfigure(encoding='utf8')
    status_file=JOB/'queue_status.json'
    prior=read(status_file) if status_file.exists() else {}
    if prior and not args.resume: raise RuntimeError('Queue exists. Inspect it before --resume.')
    if args.resume:
        if prior.get('phase') not in ('paused','failed'): raise RuntimeError('Only a paused or failed queue can resume')
        (JOB/'pause.request').unlink(missing_ok=True)
    start=time.monotonic();used=prior.get('active_seconds',0)
    def status(phase,**extra):
        write(status_file,dict(pid=os.getpid(),phase=phase,active_seconds=used+time.monotonic()-start,
                              updated_at=dt.datetime.now().astimezone().isoformat(),**extra))
    for variant in EXPERIMENTS:
        path=JOB/variant/'status.json'
        process=None
        try:
            while True:
                if (JOB/'pause.request').exists() and process is None:
                    status('paused',variant=variant,reason='Pause requested');return
                state=read(path) if path.exists() else {}
                phase=state.get('phase')
                if phase=='completed': break
                if phase in ('failed','paused') and process is not None and state.get('pid')==process.pid:
                    process.wait(timeout=60)
                    status(phase,variant=variant,child=state);return
                if phase=='failed' and not args.resume:
                    status('failed',variant=variant,child=state);return
                if phase=='paused' and not args.resume:
                    status('paused',variant=variant,child=state);return
                if (process is None and phase not in (None,'failed','paused')
                    and not process_alive(state.get('pid'))):
                    status('failed',variant=variant,child=state,reason='Training process exited without a terminal state')
                    return
                if not state or (phase in ('paused','failed') and args.resume and process is None):
                    out=(JOB/f'{variant}.queue.stdout.log').open('a' if state else 'w',encoding='utf8')
                    err=(JOB/f'{variant}.queue.stderr.log').open('a' if state else 'w',encoding='utf8')
                    cmd=[sys.executable,'-u',str(ROOT/'final_round.py'),'train','--variant',variant]
                    if state: cmd.append('--resume')
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err)
                if process is not None and process.poll() is not None:
                    state=read(path)
                    if state.get('phase')!='completed':
                        status('failed',variant=variant,child=state,exit_code=process.returncode);return
                status('running',variant=variant,child=state)
                # Reserve ten minutes for final OOF replay and inference.
                if used+time.monotonic()-start>17400:
                    (JOB/'pause.request').write_text('Round budget: pause and retain full state',encoding='utf8')
                time.sleep(10)
        finally:
            if process is not None:
                if process.poll() is None:
                    if sys.exc_info()[0] is not None:
                        (JOB/'pause.request').write_text('Queue exiting: save checkpoint',encoding='utf8')
                    process.wait(timeout=120)
                out.close();err.close()
    if (JOB/'pause.request').exists():
        status('paused',reason='Pause requested before final inference');return
    status('final_validation_and_inference')
    result=subprocess.run([sys.executable,'-u',str(ROOT/'final_round.py'),'finish'],cwd=ROOT)
    if result.returncode: status('failed',exit_code=result.returncode)
    else: status('completed',completion=read(REPORT/'completion.json'))


if __name__=='__main__': main()
