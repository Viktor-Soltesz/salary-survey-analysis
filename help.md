# Project Setup Instructions
This project consists of Jupyter notebooks, so you’ll need to have Jupyter installed along with the required libraries listed in 'requirements.txt'

## 1. Change Directory
First, navigate to the project directory where this 'help.txt' file is located. In a Terminal or Command Prompt, run:
cd /path/to/your/project

## 2. Create a virtual environment (Optional, but highly recommended):
python -m venv myenv

## 3. Activate Environment:
myenv\Scripts\activate

## 4. Install the required packages:
pip install -r requirements.txt

## 5. Install Jupyter lab:
I recommend using Jupyter lab, as it provides additional features when working with large notebooks:
pip install jupyterlab

## 6. Start jupyter lab by typing:
jupyter lab

## 7. Open and Run a Notebook:
In the Jupyter interface navigate to the notebook file you want to open.
Click on the notebook to open it.
To execute each cell, click on the cell and press Shift + Enter.

## 8. Additional Tips
Deactivate the Virtual Environment: When you’re done working, you can deactivate the environment by typing:
deactivate