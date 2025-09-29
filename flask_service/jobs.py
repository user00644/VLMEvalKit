import uuid
import json
import os
import subprocess
import threading
import tempfile
from datetime import datetime
from typing import Optional


class Job:
    def __init__(self, job_id: str, model, data, work_dir: str, proc: Optional[subprocess.Popen] = None, extra_args=None):
        self.job_id = job_id
        self.model = model
        self.data = data
        self.created_at = datetime.utcnow().isoformat() + 'Z'
        self.work_dir = work_dir
        self.proc = proc
        self.status = 'pending'
        self.extra_args = extra_args or {}

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'model': self.model,
            'data': self.data,
            'created_at': self.created_at,
            'work_dir': self.work_dir,
            'status': self.status,
            'pid': self.proc.pid if self.proc else None,
            'extra_args': self.extra_args,
        }


class JobManager:
    def __init__(self, work_root: str):
        self.work_root = work_root
        os.makedirs(self.work_root, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.meta_file = os.path.join(self.work_root, '.flask_jobs.json')
        self._load()

    def _save(self):
        try:
            with open(self.meta_file, 'w') as f:
                json.dump({k: v.to_dict() for k, v in self.jobs.items()}, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if os.path.exists(self.meta_file):
            try:
                data = json.load(open(self.meta_file))
                for k, v in data.items():
                    job = Job(job_id=v['job_id'], model=v['model'], data=v['data'], work_dir=v['work_dir'], proc=None, extra_args=v.get('extra_args'))
                    job.status = v.get('status', 'pending')
                    self.jobs[k] = job
            except Exception:
                pass

    def create_job(self, model, data, work_dir: Optional[str] = None, mode: str = 'all', extra_args: dict = None):
        job_id = uuid.uuid4().hex[:8]
        work_dir = work_dir or os.path.join(self.work_root, job_id)
        os.makedirs(work_dir, exist_ok=True)
        job = Job(job_id=job_id, model=model, data=data, work_dir=work_dir, extra_args=extra_args)
        with self.lock:
            self.jobs[job_id] = job
            self._save()

        thread = threading.Thread(target=self._run_job, args=(job, mode), daemon=True)
        thread.start()
        return job

    def _build_config_file(self, job: Job) -> Optional[str]:
        # If LMDeploy credentials provided, build a minimal config JSON that run.py can use
        extra = job.extra_args or {}
        lm_key = extra.get('lmdeploy_api_key')
        lm_base = extra.get('lmdeploy_api_base')
        lm_model = extra.get('lmdeploy_model_name')

        judge_key = extra.get('judge_api_key')
        judge_base = extra.get('judge_api_base')
        judge_model = extra.get('judge_model_name')

        if not lm_key or not lm_base:
            # No LMDeploy credentials supplied, use environment variables if available; no config written
            return None

        # Prepare a minimal config that defines one model 'lmdeploy' using class LMDeployAPI
        config = {
            "model": {
                "lmdeploy": {
                    "class": "LMDeployAPI",
                    "key": lm_key,
                    "api_base": lm_base,
                }
            },
            "data": {}
        }
        # If caller passed lm_model, attach as `model` field
        if lm_model:
            config['model']['lmdeploy']['model'] = lm_model

        # If judge credentials provided, set env var later instead
        fd, path = tempfile.mkstemp(prefix='vlmeval_config_', suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f)
        return path


    def _run_job(self, job: Job, mode: str):
        # Build command to run VLMEvalKit's run.py
        cmd = ["python", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'run.py'))]

        # Pass datasets (data can be list or str)
        if isinstance(job.data, (list, tuple)):
            cmd += ["--data"] + list(job.data)
        else:
            cmd += ["--data", job.data]

        # We use `lmdeploy` as the model name
        cmd += ["--model", job.model]
        cmd += ["--work-dir", job.work_dir]
        cmd += ["--reuse"]

        # Support dry-run mode: if extra_args contains dry_run=True, create fake outputs and finish quickly
        extra = job.extra_args or {}
        if extra.get('dry_run'):
            job.status = 'running'
            self._save()
            # create per-run log even for dry-run
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            per_run_log = os.path.join(job.work_dir, f'vlmeval_{ts}.log')
            job_log_path = os.path.join(job.work_dir, 'job.log')
            svc_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flask_service.log'))
            try:
                with open(per_run_log, 'w') as run_out, open(job_log_path, 'w') as summary, open(svc_log_path, 'a') as svc_out:
                    summary.write(f'Per-run log: {os.path.basename(per_run_log)}\n')
                    summary.write(f'Created at: {ts}\n')
                    # simulate both plain and TTY-style output for verification
                    run_out.write('Dry-run: no real evaluation performed.\n')
                    run_out.write(f'Model: {job.model}\n')
                    run_out.write(f'Data: {job.data}\n')
                    run_out.write('Writing fake prediction file...\n')
                    # Simulated rich/tty output
                    run_out.write('\x1b[1;32m[SIMULATED-TTY] Progress: 100%\x1b[0m\n')
                    svc_out.write(f'[DRYRUN {ts}] Job {job.job_id}: Dry-run performed\n')

                # Create a fake prediction file (simple JSON)
                fake = {
                    'index': [0],
                    'prediction': ['This is a dry-run fake prediction.']
                }
                import json as _json
                with open(os.path.join(job.work_dir, 'fake_prediction.json'), 'w') as f:
                    _json.dump(fake, f, indent=2)

                job.status = 'finished'
            except Exception as e:
                job.status = 'failed'
                try:
                    with open(job_log_path, 'a') as out:
                        out.write(str(e))
                except Exception:
                    pass
            finally:
                self._save()
            return

        # # Prepare environment for subprocess
        # env = os.environ.copy()

        # # Inject LMDeploy and judge credentials into env if provided
        # extra = job.extra_args or {}
        # if extra.get('lmdeploy_api_key'):
        #     env['LMDEPLOY_API_KEY'] = extra.get('lmdeploy_api_key')
        # if extra.get('lmdeploy_api_base'):
        #     env['LMDEPLOY_API_BASE'] = extra.get('lmdeploy_api_base')
        # if extra.get('lmdeploy_model_name'):
        #     env['LMDEPLOY_MODEL_NAME'] = extra.get('lmdeploy_model_name')

        # # Judge / openai style environment variables
        # if extra.get('judge_api_key'):
        #     env['OPENAI_API_KEY'] = extra.get('judge_api_key')
        # if extra.get('judge_api_base'):
        #     env['OPENAI_API_BASE'] = extra.get('judge_api_base')
        # if extra.get('judge_model_name'):
        #     env['LOCAL_LLM'] = extra.get('judge_model_name')
        # if extra.get('LMUData'):
        #     env['LMUData'] = extra.get('LMUData')

        job.status = 'running'
        self._save()
        # job.log remains a short summary file; per-run detailed log is saved separately
        job_log_path = os.path.join(job.work_dir, 'job.log')
        # central service log at repository root
        svc_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'flask_service.log'))

        # per-run log with UTC timestamp
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        per_run_log = os.path.join(job.work_dir, f'vlmeval_{ts}.log')

        try:
            # write header to job.log indicating the per-run log file
            with open(job_log_path, 'w') as summary:
                summary.write(f'Per-run log: {os.path.basename(per_run_log)}\n')
                summary.write(f'Created at: {ts}\n')

            with open(per_run_log, 'w') as run_out, open(svc_log_path, 'a') as svc_out:
                def log_all(msg):
                    """Helper to write log messages to both per-run and central service logs."""
                    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    # To per-run log
                    run_out.write(f"[{timestamp}] {msg}\n")
                    run_out.flush()
                    # To central service log, prefixed with job_id for context
                    svc_out.write(f"[{timestamp}] [{job.job_id}] {msg}\n")
                    svc_out.flush()

                log_all(f"Job starting. Mode: {mode}. Work dir: {job.work_dir}")

                try:
                    use_pty = extra.get('use_pty', True)

                    # Dry-run branch
                    if extra.get('dry_run'):
                        log_all("[INFO] Dry-run mode enabled. No real evaluation will be performed.")
                        run_out.write('Dry-run: no real evaluation performed.\n')
                        run_out.write(f'Model: {job.model}\n')
                        run_out.write(f'Data: {job.data}\n')
                        run_out.write('Writing fake prediction file...\n')
                        if use_pty:
                            run_out.write('\x1b[1;32m[SIMULATED-TTY] Progress: 100%\x1b[0m\n')
                        
                        fake = {'index': [0], 'prediction': ['This is a dry-run fake prediction.']}
                        with open(os.path.join(job.work_dir, 'fake_prediction.json'), 'w') as f:
                            json.dump(fake, f, indent=2)
                        
                        job.status = 'finished'
                        log_all("[INFO] Dry-run finished successfully.")
                        self._save()
                        return

                    # Real-run branch
                    log_all("[INFO] Real run mode. Preparing to execute VLMEvalKit.")

                    # Prepare environment for subprocess
                    env = os.environ.copy()
                    env_vars_set = []

                    if extra.get('lmdeploy_api_key'):
                        env['LMDEPLOY_API_KEY'] = extra.get('lmdeploy_api_key')
                        env_vars_set.append('LMDEPLOY_API_KEY')
                    if extra.get('lmdeploy_api_base'):
                        env['LMDEPLOY_API_BASE'] = extra.get('lmdeploy_api_base')
                        env_vars_set.append('LMDEPLOY_API_BASE')
                    if extra.get('lmdeploy_model_name'):
                        env['LMDEPLOY_MODEL_NAME'] = extra.get('lmdeploy_model_name')
                        env_vars_set.append('LMDEPLOY_MODEL_NAME')

                    if extra.get('judge_api_key'):
                        env['OPENAI_API_KEY'] = extra.get('judge_api_key')
                        env_vars_set.append('OPENAI_API_KEY')
                    if extra.get('judge_api_base'):
                        env['OPENAI_API_BASE'] = extra.get('judge_api_base')
                        env_vars_set.append('OPENAI_API_BASE')
                    if extra.get('judge_model_name'):
                        env['LOCAL_LLM'] = extra.get('judge_model_name')
                        env_vars_set.append('LOCAL_LLM')
                    if extra.get('LMUData'):
                        env['LMUData'] = extra.get('LMUData')
                        env_vars_set.append('LMUData')  
                    
                    log_all(f"[DEBUG] Environment variables set for subprocess: {', '.join(env_vars_set) if env_vars_set else 'None'}")

                    log_all(f"[INFO] Executing command: {' '.join(cmd)}")

                    if use_pty:
                        log_all("[DEBUG] Using PTY for subprocess execution.")
                        import pty
                        master_fd, slave_fd = pty.openpty()
                        proc = subprocess.Popen(
                            cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                            cwd=os.getcwd(), env=env, close_fds=True,
                        )
                        job.proc = proc
                        os.close(slave_fd)
                        try:
                            while True:
                                try:
                                    data = os.read(master_fd, 1024)
                                except OSError: break
                                if not data: break
                                text = data.decode('utf-8', errors='replace')
                                run_out.write(text)
                                run_out.flush()
                                svc_out.write(text) # Raw output to central log
                                svc_out.flush()
                        finally:
                            os.close(master_fd)
                    else:
                        log_all("[DEBUG] Using PIPE for subprocess execution (no TTY).")
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=os.getcwd(), env=env,
                        )
                        job.proc = proc
                        for line in proc.stdout:
                            log_all(f"[RUN] {line.strip()}") # Log each line with a prefix

                    proc.wait()
                    job.status = 'finished' if proc.returncode == 0 else 'failed'
                    log_all(f"[INFO] Command finished with exit code: {proc.returncode}. Job status: {job.status}")

                except Exception as e:
                    job.status = 'failed'
                    log_all(f"[ERROR] Job failed with exception: {e}")
                finally:
                    if 'cfg_path' in locals() and cfg_path and os.path.exists(cfg_path):
                        try:
                            os.remove(cfg_path)
                            log_all(f"[DEBUG] Cleaned up temporary config file: {cfg_path}")
                        except Exception: pass
                    self._save()
        except Exception:
            # Fallback: simple invocation writing to job.log if per-run log cannot be created
            try:
                with open(job_log_path, 'a') as out:
                    proc = subprocess.Popen(cmd, stdout=out, stderr=out, cwd=os.getcwd(), env=env)
                    job.proc = proc
                    proc.wait()
                    job.status = 'finished' if proc.returncode == 0 else 'failed'
            except Exception as e:
                job.status = 'failed'
                try:
                    with open(job_log_path, 'a') as out:
                        out.write(str(e))
                except Exception:
                    pass
            finally:
                if cfg_path and os.path.exists(cfg_path):
                    try:
                        os.remove(cfg_path)
                    except Exception:
                        pass
                self._save()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)
