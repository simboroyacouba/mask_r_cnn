import os

def generate_tree(path, prefix="", max_files=3):
    entries = sorted(os.listdir(path))
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    
    all_entries = dirs + files
    total = len(all_entries)
    
    for i, entry in enumerate(all_entries):
        is_last = (i == total - 1)
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        full_path = os.path.join(path, entry)
        
        if os.path.isdir(full_path):
            print(f"{prefix}{connector}{entry}/")
            generate_tree(full_path, prefix + extension, max_files)
        else:
            # Limiter le nombre de fichiers affichés
            file_index = files.index(entry)
            if file_index < max_files:
                print(f"{prefix}{connector}{entry}")
            elif file_index == max_files:
                print(f"{prefix}{'└── ' if is_last else '├── '}...")
                break

def print_project_tree(project_path, max_files=3):
    project_name = os.path.basename(os.path.abspath(project_path))
    print(f"{project_name}/")
    generate_tree(project_path, prefix="", max_files=max_files)

# Utilisation
project_path = "."  # Chemin vers ton dossier projet
print_project_tree(project_path, max_files=3)