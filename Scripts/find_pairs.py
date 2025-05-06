import argparse
import os
import shutil


def find_unpaired(dir_a: str, dir_b: str, dir_source: str):
    unpaired = []
    copy_counter = 0
    for entry in os.listdir(dir_a):
        path_a = os.path.join(dir_a, entry)
        if not os.path.isfile(path_a):
            continue

        base, ext = os.path.splitext(entry)
        ext = ext.lower()
        if ext not in {'.png', '.json'}:
            continue

        other_ext = '.json' if ext == '.png' else '.png'
        counterpart = base + other_ext
        path_b = os.path.join(dir_b, counterpart)

        if not os.path.isfile(path_b):
            # Check if the counterpart exists in the source directory
            path_source = os.path.join(dir_source, counterpart)
            if os.path.isfile(path_source):
                shutil.copy2(path_source, path_b)
                copy_counter += 1
            else:
                unpaired.append(entry)

    return unpaired, copy_counter


def main():
    parser = argparse.ArgumentParser(
        description="List .png/.json files in dir_a that have no matching counterpart in dir_b."
    )
    parser.add_argument('dir_a', help="Directory to scan for .png/.json files")
    parser.add_argument('dir_b', help="Directory in which to look for matching counterparts")
    parser.add_argument('source', help="Directory to pull missing files from")
    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)
    dir_source = os.path.abspath(args.source)

    if not os.path.isdir(dir_a):
        print(f"Error: {dir_a} is not a directory.")
        return
    if not os.path.isdir(dir_b):
        print(f"Error: {dir_b} is not a directory.")
        return

    missing, copy_counter = find_unpaired(dir_a, dir_b, dir_source)
    if not missing:
        print(f"All .png and .json files in '{dir_a}' have matching counterparts in '{dir_b}'.")
        return

    for filename in missing:
        print(filename)
        
    print(f"Total missing files: {len(missing)}")
    print(f"Total files copied: {copy_counter}")


if __name__ == "__main__":
    main()
