"""
Script to execute the Jupyter Notebook programmatically using nbclient.
Executes all cells, populates outputs and inline plots, and saves the notebook inplace.
"""
import nbformat
from nbclient import NotebookClient

nb_path = "bar_inventory_project/notebooks/inventory_forecasting_solution.ipynb"

print(f"Loading notebook from {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")

print("Executing notebook cells...")
try:
    client.execute()
    print("Execution completed successfully!")
except Exception as e:
    print(f"Notebook execution failed: {e}")

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully saved executed notebook to {nb_path}")
