import os
import platform
import numpy as np


def _cf_real_fit_model(tlist, a, b, c, d=0):
    return np.real((a + 1j * d) * np.exp((b + 1j * c) * np.abs(tlist)))


def _cf_imag_fit_model(tlist, a, b, c, d=0):
    return np.sign(tlist) * np.imag(
        (a + 1j * d) * np.exp((b + 1j * c) * np.abs(tlist))
    )


def _unpack(params, num_params):
    N = len(params) // num_params
    return np.reshape(params, (N, num_params))


def _evaluate(fun, x, params):
    result = 0
    for term_params in params:
        result += fun(x, *term_params)
    return result


def print_full_precision(name, arr):
    print(f"\n[{name}] Shape: {arr.shape}")
    # Print formatted standard decimal (17 digits preserves exact 64-bit float)
    formatted_dec = np.array2string(
        arr, precision=17, separator=", ", floatmode="unique"
    )
    print(f"Decimal (max precision):\n{formatted_dec}")


def main():
    import scipy
    from scipy.optimize import curve_fit

    # Hardware & Environment Diagnostics
    cpu_arch = platform.machine()
    proc_desc = platform.processor() or "Unknown"
    env_override = os.environ.get("OPENBLAS_CORETYPE", "None (Auto-detected)")
    config = scipy.show_config(mode='dicts')
    blas_info = config.get('Build Dependencies', {}).get('blas', {})

    print("=" * 80)
    print(f"Host Machine Architecture : {cpu_arch} ({proc_desc})")
    print(f"Operating System          : {platform.system()} {platform.release()}")
    print(f"Requested Target Override : {env_override}")
    print(f"OpenBLAS Dispatched Core  : {blas_info['openblas configuration']}")
    print(f"SciPy Version             : {scipy.__version__}")
    print(f"NumPy Version             : {np.__version__}")
    print("=" * 80)

    # Use paths relative to current script/repository directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(script_dir, "fit_real.npz")
    imag_path = os.path.join(script_dir, "fit_imag.npz")

    if not os.path.exists(real_path) or not os.path.exists(imag_path):
        raise FileNotFoundError(
            f"Missing test data! Ensure fit_real.npz and fit_imag.npz exist in {script_dir}"
        )

    data_real = np.load(real_path, allow_pickle=True)
    data_imag = np.load(imag_path, allow_pickle=True)

    num_params_r = 3
    Ni_min_r = 2
    guess_r = [0.07129573917816213, -0.4, 0.0]
    lower_r = [-1.4259147835632426, -np.inf, 0.0]
    upper_r = [1.4259147835632426, 0.1, np.inf]

    num_params_ig = 3
    Ni_min_ig = 1
    guess_ig = [-0.04180271993678641, -0.4180271993678641, 1.0]
    lower_ig = [-0.8360544, -np.inf, 0.0]
    upper_ig = [0.4180272, 0.0, np.inf]

    # Real Fit
    packed_params, _ = curve_fit(
        lambda x, *p: _evaluate(
            _cf_real_fit_model, x, _unpack(p, num_params_r)
        ),
        data_real["tlist"],
        data_real["clist_real"],
        p0=np.tile(guess_r, (Ni_min_r, 1)).ravel(),
        bounds=(np.tile(lower_r, Ni_min_r), np.tile(upper_r, Ni_min_r)),
    )

    # Imaginary Fit
    packed_params_ig, _ = curve_fit(
        lambda x, *p: _evaluate(
            _cf_imag_fit_model, x, _unpack(p, num_params_ig)
        ),
        data_imag["tlist"],
        data_imag["clist_imag"],
        p0=np.tile(guess_ig, (Ni_min_ig, 1)).ravel(),
        bounds=(np.tile(lower_ig, Ni_min_ig), np.tile(upper_ig, Ni_min_ig)),
    )

    fit_real_res = _unpack(packed_params, num_params_r)
    fit_imag_res = _unpack(packed_params_ig, num_params_ig)

    print_full_precision("Real Fit Output", fit_real_res)
    print_full_precision("Imag Fit Output", fit_imag_res)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
