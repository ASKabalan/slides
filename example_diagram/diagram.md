# AI Assistant Instructions for Diagram Generation

## 1. Central Generator File

All Python code for generating diagrams must be located in a single file: `assets/diagram_generator.py`. If this file or the `assets` directory does not exist, you must create it.

---

## 2. Dedicated Functions

For each new diagram requested, you must add a corresponding function to the `diagram_generator.py` file. This function will contain all the necessary code to generate that specific diagram.

---

## 3. Command-Line Controls

The `diagram_generator.py` script must be runnable from the command line and include argument parsing using Python's `argparse` library. Your implementation should be similar in structure to the `example_diagram/diagram.py` file provided previously.

### Flags:

* **-a, --all**: When this flag is used, the script will execute **all** diagram-generating functions within the file.
* **-f, --functions**: This flag accepts a comma-separated string of function names. When used, the script should **only** execute the functions specified in that string.