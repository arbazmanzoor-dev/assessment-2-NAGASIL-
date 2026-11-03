# Debug log

Machine: macOS, Apple Silicon (arm64), CPU only. No conda. System Python 3.14.

| # | Stage | Error / symptom | Root cause | Fix |
|---|-------|-----------------|------------|-----|
| 1 | Environment | Original setup needs Python 3.6 + `tensorflow==1.13` (+ `tensorflow-gpu`, `intel-numpy`, `mujoco-py`, ...) | TF 1.x has no wheels for Apple Silicon or modern Python; `requirements.txt` pins ~120 packages, most unused | Did **not** use `requirements.txt`. Grepped every `import` in the scripts we run and installed only those: `uv venv -p 3.11` + `tensorflow==2.15.1 "numpy<2" scipy networkx==2.8.8 matplotlib gym==0.26.2 tqdm pandas`. TF 2.15 is the last release where `tf.compat.v1.layers` works (Keras 3 arrives in 2.16). `numpy<2` because TF 2.15 is built against NumPy 1.x. `networkx` 2.x so the authors' pickled graphs still load. |
| 2 | Environment | `uv pip install` showed no progress for ~11 min (7 s CPU) | Not a hang. The 196 MB TF wheel was downloading at ~0.8 MB/s and `uv` prints nothing while downloading | Ran with `-v` to see it was on `Downloading tensorflow (196.2MiB)`, checked speed with `curl -w %{speed_download}`, then re-ran the install in the background |
| 3 | Smoke test (`bprun_random.py`) | `AttributeError: module 'tensorflow' has no attribute 'ConfigProto'` | TF1 API (`ConfigProto`, `Session`, `set_random_seed`, `placeholder`, `layers.dense`, `log`, `multinomial`) moved under `tf.compat.v1` in TF2 | Replaced `import tensorflow as tf` with `import tensorflow.compat.v1 as tf` + `tf.disable_v2_behavior()` in `bprun_{random,maxinf,maxdef,gasil}.py`, `algo/ppo.py`, `network_models/policy_net.py`, `network_models/discriminator.py`. `tf.contrib` was never used, so nothing else needed changing |
| 4 | GASIL (fixed before it could crash) | `np.asscalar` is called 3 times in `bprun_gasil.py` | `np.asscalar` was removed in NumPy 1.23 (`AttributeError` at the first env step) | `np.asscalar(x)` → `np.array(x).item()`; `np.asscalar(np.array([picked_act]))` → `int(picked_act)` |
| 5 | Real runs | `bprun_gasil.py: error: unrecognized arguments: --logdir log/gasil_250` | My launcher loop kept `extra="--logdir log/gasil_250"` in a variable. **zsh does not word-split unquoted variables** (bash does), so argparse got one argument containing a space | Ran the GASIL command directly with explicit arguments |
| 6 | Plotting | `UserWarning: loadtxt: input contained no data` for `result/bG_1250_08/*` | The run scripts write results through a buffered file that only flushes on `close()`, so files of runs still going are empty | `plot_results.py` skips empty result files |

Warnings left as they are (harmless):
- `gym` prints "Gym has been unmaintained since 2022..." on import. Only `gym.spaces.Discrete` is used.
- `FutureWarning: adjacency_matrix will return a scipy.sparse array` in `EoN/simulation.py`, from networkx 2.8.8.
- TF deprecation warnings about non-resource variables, from `disable_v2_behavior()`.

No changes were needed to `env/epidemic.py`, `EoN/`, or `prun_util.py`. Their networkx usage (`G.nodes[...]`, `adjacency_matrix`) still works on networkx 2.8.8.
