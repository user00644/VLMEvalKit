from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from .jobs import JobManager
import os

app = Flask(__name__)

# Configure work root (where VLMEvalKit outputs results)
WORK_ROOT = os.environ.get("MMEVAL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")))
os.makedirs(WORK_ROOT, exist_ok=True)

manager = JobManager(work_root=WORK_ROOT)


@app.route("/submit", methods=["POST"])
def submit():
    """Submit an evaluation job.

    Expected JSON body keys (supports multiple modes):
    - lmdeploy_api_key, lmdeploy_api_base, lmdeploy_model_name  (for evaluation model)
    - judge_api_key, judge_api_base, judge_model_name, local_llm  (for judge model; optional)
    - data: dataset name or list of dataset names (required)
    - work_dir: optional subdirectory under outputs
    - mode: 'all'|'infer'|'eval'
    - extra_args: dict mapping to extra command line args
    """
    spec = request.get_json() or {}

    # Collect LMDeploy evaluation model credentials
    lmdeploy_api_key = spec.get('lmdeploy_api_key')
    lmdeploy_api_base = spec.get('lmdeploy_api_base')
    lmdeploy_model_name = spec.get('lmdeploy_model_name')

    # Judge model credentials (optional)
    judge_api_key = spec.get('judge_api_key')
    judge_api_base = spec.get('judge_api_base')
    judge_model_name = spec.get('judge_model_name')
    LMUData = spec.get('LMUData')

    data = spec.get('data')
    if not data:
        return jsonify({"error": "`data` (dataset name or list) is required"}), 400

    work_dir = spec.get("work_dir")
    mode = spec.get("mode", "all")
    extra = spec.get("extra_args", {}) or {}

    # Consolidate credentials into extra so JobManager can set env and config
    extra.update({
        'lmdeploy_api_key': lmdeploy_api_key,
        'lmdeploy_api_base': lmdeploy_api_base,
        'lmdeploy_model_name': lmdeploy_model_name,
        'judge_api_key': judge_api_key,
        'judge_api_base': judge_api_base,
        'judge_model_name': judge_model_name,
        'LMUData': LMUData,
    })

    # We will use `lmdeploy` as the model entry in VLMEvalKit config to invoke the LMDeploy API
    model = 'lmdeploy'

    job = manager.create_job(model=model, data=data, work_dir=work_dir, mode=mode, extra_args=extra)
    return jsonify({"job_id": job.job_id, "status": job.status})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    j = manager.get_job(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404
    return jsonify(j.to_dict())


@app.route("/results/<job_id>", methods=["GET"])
def results(job_id):
    j = manager.get_job(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404
    if j.status != "finished":
        return jsonify({"error": "job not finished", "status": j.status}), 400
    out_dir = j.work_dir
    files = []
    for root, _, filenames in os.walk(out_dir):
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(root, fn), out_dir))
    return jsonify({"work_dir": out_dir, "files": files})


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    path = request.args.get("path")
    j = manager.get_job(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404
    if j.status != "finished":
        return jsonify({"error": "job not finished", "status": j.status}), 400
    if not path:
        return jsonify({"error": "query param 'path' required (relative to work_dir)"}), 400
    file_path = os.path.join(j.work_dir, path)
    if not os.path.exists(file_path):
        return jsonify({"error": "file not found"}), 404
    return send_file(file_path, as_attachment=True)


def _select_log_file(job, path: str = None):
    # If a specific relative path is requested, use it; otherwise pick latest vlmeval_*.log or job.log
    if path:
        candidate = os.path.join(job.work_dir, path)
        if os.path.exists(candidate):
            return candidate
        return None

    # pick latest vlmeval_*.log
    logs = [os.path.join(job.work_dir, f) for f in os.listdir(job.work_dir) if f.startswith('vlmeval_') and f.endswith('.log')]
    if logs:
        logs.sort()
        return logs[-1]

    # fallback to job.log
    jl = os.path.join(job.work_dir, 'job.log')
    if os.path.exists(jl):
        return jl
    return None


@app.route('/logs/<job_id>', methods=['GET'])
def logs_tail(job_id):
    """Return the last N lines of a per-job log file.

    Query params:
    - path: optional relative path under job work_dir (e.g. vlmeval_2025... .log or job.log)
    - tail: number of lines to return (default 200)
    """
    j = manager.get_job(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404

    rel_path = request.args.get('path')
    tail = int(request.args.get('tail') or 200)
    file_path = _select_log_file(j, rel_path)
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "log file not found"}), 404

    # Read last N lines efficiently
    try:
        with open(file_path, 'rb') as f:
            # Seek from end in blocks
            avg_line_len = 200
            to_read = tail * avg_line_len
            try:
                f.seek(-to_read, os.SEEK_END)
            except OSError:
                f.seek(0)
            data = f.read().decode('utf-8', errors='replace')
        lines = data.splitlines()[-tail:]
        return Response("\n".join(lines) + "\n", mimetype='text/plain')
    except Exception as e:
        return jsonify({"error": "failed to read log", "detail": str(e)}), 500


@app.route('/logs/<job_id>/stream', methods=['GET'])
def logs_stream(job_id):
    """Stream live updates of a job log file (simple tail -f style).

    Query params:
    - path: optional relative path under job work_dir
    - timeout: seconds to stream (default 60)
    """
    j = manager.get_job(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404

    rel_path = request.args.get('path')
    timeout = int(request.args.get('timeout') or 60)
    file_path = _select_log_file(j, rel_path)
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "log file not found"}), 404

    def generator(path, timeout_sec):
        import time
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                # Seek to end
                fh.seek(0, os.SEEK_END)
                start = time.time()
                while True:
                    line = fh.readline()
                    if line:
                        yield line
                    else:
                        time.sleep(0.3)
                    if time.time() - start > timeout_sec:
                        break
        except Exception as e:
            yield f"[stream-error] {e}\n"

    return Response(stream_with_context(generator(file_path, timeout)), mimetype='text/plain')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
