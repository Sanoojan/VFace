import os
import yaml
import argparse
from natsort import natsorted

def create_matching_yaml(subfolders_root, source_images_root, output_yaml_path):
    # Get sorted list of subfolders and source images
    subfolders = natsorted([f for f in os.listdir(subfolders_root) if os.path.isdir(os.path.join(subfolders_root, f))])
    source_images = natsorted([f for f in os.listdir(source_images_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(subfolders) == len(source_images), "Mismatch between subfolders and source images!"

    # Build the matching dictionary
    matching = {subfolder: image_name for subfolder, image_name in zip(subfolders, source_images)}

    # Save to a single YAML file
    with open(output_yaml_path, 'w') as f:
        yaml.dump(matching, f)

    print(f"Matching YAML saved to {output_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a YAML file matching subfolders to source images.")
    parser.add_argument('--video_base_dir', type=str, required=True, help='Path to the folder containing subfolders')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the folder containing source images')
    parser.add_argument('--output_yaml_path', type=str, required=True, help='Path to save the output YAML file')

    args = parser.parse_args()

    create_matching_yaml(
        subfolders_root=args.video_base_dir,
        source_images_root=args.image_dir,
        output_yaml_path=args.output_yaml_path
    )
