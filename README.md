# Refactored pyOMA2 Algorithms

This repository contains a refactored, data-only version of the core algorithms from the `pyOMA2` library. The goal of this refactoring was to create a lightweight, stateless, and non-GUI version of the algorithms suitable for scripting and automated data processing pipelines.

## Summary of Changes

The primary changes from the original `pyOMA2` library are:

-   **Stateless Design**: Algorithm instances no longer store results internally. Methods like `run()` and `mpe()` now return result objects directly.
-   **No GUI/Plotting**: All dependencies on `matplotlib`, `pyvista`, and other plotting/GUI libraries have been removed from the core algorithm classes.
-   **Data-Focused API**: The API is designed for a clear data-in, data-out workflow.

For a detailed breakdown of the changes, please see the `refactoring_summary.md` file that will also be included in this repository.

## How to Use

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Use in your script**:
    The following example demonstrates how to use the `FDD` algorithm. The same pattern applies to `EFDD`, `FSDD`, `pLSCF`, and `SSI`.

    ```python
    import numpy as np
    from pyoma2.algorithms import FDD
    # You would typically load your own data here
    from pyoma2.functions.gen import example_data

    # 1. Get your data and sampling frequency
    Y, U, modal_params = example_data()
    fs = 100.0

    # 2. Instantiate the algorithm with parameters
    fdd_algo = FDD(nxseg=1024, method_SD='per', pov=0.5)

    # 3. Provide the data to the algorithm instance
    fdd_algo._set_data(data=Y, fs=fs)

    # 4. Run the analysis to get initial results
    run_result = fdd_algo.run()

    # 5. Run Modal Parameter Estimation (MPE)
    sel_freq = [0.8, 2.5, 4.0, 5.2, 6.0]
    final_result = fdd_algo.mpe(run_result, sel_freq, DF=0.2)

    # 6. Access the results
    print("Extracted Frequencies (Hz):")
    print(final_result.Fn)
    ```

## License

This code is provided under the original MIT License. Please see the `LICENSE` file for full details.

