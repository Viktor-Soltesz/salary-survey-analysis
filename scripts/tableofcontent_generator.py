import nbformat
import re

def generate_toc(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Initialize tracking variables
    toc = []
    counters = [0, 0, 0, 0, 0]  # Counters for up to 5 levels of headings
    
    # Process each cell
    for cell in nb.cells:
        # Only process markdown cells
        if cell['cell_type'] == 'markdown':
            # Split the cell source into lines
            lines = cell['source'].split('\n')
            
            # Check for heading lines
            for line in lines:
                line = line.strip()
                if line.startswith('#') and ' ' in line:
                    # Count heading level
                    level = line.count('#', 0, line.find(' '))
                    
                    # Reset lower-level counters
                    for i in range(level, len(counters)):
                        counters[i] = 0
                    
                    # Increment counter for this level
                    counters[level - 1] += 1
                    
                    # Create numbering string
                    numbering = '.'.join(map(str, counters[:level]))
                    
                    # Extract title
                    title = line.split(' ', 1)[1]
                    
                    # Create TOC entry with proper indentation and numbering
                    toc.append(f"{' ' * (level * 4)} {numbering} {title}")
    
    # Return the TOC as a string
    return "\n".join(toc)

def generate_toc_withanchors(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Initialize tracking variables
    toc = []
    counters = [0, 0, 0, 0, 0]  # Counters for up to 5 levels of headings
    
    # Process each cell
    for cell in nb.cells:
        # Only process markdown cells
        if cell['cell_type'] == 'markdown':
            # Split the cell source into lines
            lines = cell['source'].split('\n')
            
            # Check for heading lines
            for line in lines:
                line = line.strip()
                if line.startswith('#') and ' ' in line:
                    # Count heading level
                    level = line.count('#', 0, line.find(' '))
                    
                    # Reset lower-level counters
                    for i in range(level, len(counters)):
                        counters[i] = 0
                    
                    # Increment counter for this level
                    counters[level - 1] += 1
                    
                    # Create numbering string
                    numbering = '.'.join(map(str, counters[:level]))
                    
                    # Extract title
                    title = line.split(' ', 1)[1]
                    
                    # Create anchor (GitHub-style)
                    anchor = re.sub(r'[^\w\s-]', '', title.lower())  # Remove non-word characters
                    anchor = re.sub(r'\s+', '-', anchor)  # Replace spaces with hyphens
                    
                    # Create TOC entry with proper indentation, numbering, and anchor
                    toc.append(f"{' ' * (level * 2)}[{numbering} {title}](#{anchor})")
    
    # Return the TOC as a string
    return "\n".join(toc)