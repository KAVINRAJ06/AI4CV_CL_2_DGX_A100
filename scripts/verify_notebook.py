"""Execute the notebook in the calling Python environment; save a local executed copy."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

source = Path("notebooks/TQSI_Training.ipynb")
notebook = nbformat.read(source, as_version=4)
nbformat.validate(notebook)
client = NotebookClient(notebook, timeout=1200, resources={"metadata": {"path": str(Path.cwd())}})
client.km = KernelManager(kernel_name="python3")
client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
try:
    client.execute()
finally:
    if client.km is not None and client.km.has_kernel:
        client.km.shutdown_kernel(now=True)
path = Path("outputs/notebook_executed.ipynb")
path.parent.mkdir(exist_ok=True)
nbformat.write(notebook, path)
print(f"Executed all notebook cells: {path}")
