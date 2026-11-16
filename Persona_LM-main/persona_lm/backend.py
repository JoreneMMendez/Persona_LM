"""Which compute backend to use. This file never imports PyTorch, so it is safe everywhere (including the embeddable runtime)."""
import importlib.util

NO_TORCH_MSG = ("PyTorch is not installed. For a CUDA GPU on your own computer, install it with the command that the selector at "
                "https://pytorch.org/get-started/locally/ gives for your system (it depends on your CUDA version). "
                "Google Colab and Kaggle already have it. Without PyTorch only the slow NumPy CPU backend is available.")


def torch_installed():
    return importlib.util.find_spec("torch") is not None


def choose_backend(want="auto", log=print):
    """want: "auto" (PyTorch if installed, else NumPy), "torch" or "numpy". Returns "torch" or "numpy"."""
    if want == "numpy":
        return "numpy"
    if want == "torch":
        if not torch_installed():
            raise SystemExit(NO_TORCH_MSG)
        return "torch"
    if want != "auto":
        raise ValueError("unknown backend " + repr(want))
    if torch_installed():
        return "torch"
    log("PyTorch not found: falling back to the slow NumPy CPU backend. (%s)" % NO_TORCH_MSG)
    return "numpy"
