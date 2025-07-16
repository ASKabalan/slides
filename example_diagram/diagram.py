#
# This script is an example demonstrating how to generate a "diagram as code"
# using the Python `diagrams` library. It is designed to serve as a reference
# or inspiration for an AI assistant like Gemini or Claude.
#
# --- Key Concepts Demonstrated ---
#
# 1. Importing a custom image (e.g., a PNG file):
#    To use a custom image, the `Custom` node from the `diagrams.custom` module is used.
#    It requires a label for the node and the file path to the image.
#    Example:
#    from diagrams.custom import Custom
#    my_node = Custom("My Custom Node", "./path/to/your/image.png")
#
# 2. Using Programming-Specific Nodes:
#    The `diagrams` library has a rich set of nodes for programming concepts,
#    which can be found in the official documentation:
#    https://diagrams.mingrammer.com/docs/nodes/programming
#    To use them, import the specific node from its module, such as
#    `diagrams.programming.flowchart` or `diagrams.programming.language`.
#    Example:
#    from diagrams.programming import flowchart, language
#    start_node = flowchart.Action("Start Process")
#    code_node = language.Python("my_script.py")
#
# 3. Generating the Diagram Image:
#    The `Diagram` object acts as a context manager. When the `with` block is exited,
#    it automatically renders the defined diagram and saves it as an image file.
#    The output filename is set using the `filename` parameter.
#    Note: Graphviz must be installed on the system for rendering to succeed.
#

from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.programming import flowchart, language
import os

# 1. Define graph-level attributes for styling
graph_attr = {
    "fontsize": "20",
    "bgcolor": "transparent",
    "compound": "true",
    "splines": "spline",
}

# Get the directory of the current script to build absolute paths
# This makes the script runnable from anywhere
current_file_path = os.path.dirname(os.path.abspath(__file__))


output_filename = os.path.join(current_file_path, "scientific_concept_flow")
cmb_image_path = os.path.join(current_file_path, "resources/CMB-Planck.png")

# 2. Create the diagram context
with Diagram("Scientific Concept Flow", show=False, filename=output_filename, graph_attr=graph_attr):

    # Start with the custom CMB image
    cmb_data = Custom("Raw CMB Data", cmb_image_path)

    # The data leads to a scientific concept (a document)
    concept = flowchart.Document("Scientific\nConcept")

    # The concept is used to formulate a hypothesis (a triangle/decision node)
    hypothesis = flowchart.Decision("Formulate\nHypothesis")

    # The hypothesis is tested with code, leading to scientific analysis in a box
    with Cluster("Analysis"):
        analysis_script = language.Python("Analysis Script")
        science_box = flowchart.PredefinedProcess("Scientific Results")
        analysis_script >> science_box


    # 3. Connect the nodes in the requested sequence
    cmb_data >> Edge(label="inspires") >> concept
    concept >> Edge(label="leads to") >> hypothesis
    hypothesis >> Edge(label="is tested by") >> analysis_script

