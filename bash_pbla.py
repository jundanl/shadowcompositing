import os
import glob
import argparse


def parse_argument():
    parser = argparse.ArgumentParser(description='Bash script for running inference on images using the Shadow Harmonization Network (pbla)')
    parser.add_argument("--script_path", type=str, default="./src/network/test_pbla.py")
    parser.add_argument("--input_folder", type=str, default="./test_data/collect_ours_userstudy")
    parser.add_argument("--image_filter", type=str, default="_background.png")
    parser.add_argument("--split_obj_name", type=str, default="_Sphere")
    parser.add_argument("--output_folder", type=str, default="./out/")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default='cuda:0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_argument()
    assert os.path.exists(args.script_path), f"Script path: {args.script_path} does not exist"
    assert os.path.exists(args.input_folder), f"Input folder: {args.input_folder} does not exist"
    args.output_folder = os.path.join(args.output_folder, os.path.basename(args.input_folder))
    print(F"Save output to {args.output_folder} ......")

    # Get image list
    img_list = glob.glob(os.path.join(args.input_folder, "*" + args.image_filter))
    img_list = sorted(img_list)
    print(f"Found {len(img_list)} images in {args.input_folder} with filter {args.image_filter}.")

    # Run inference
    processed_set = set()
    for i in range(0, len(img_list)):
        # Image file paths
        img_full_path = img_list[i]
        img_file_path = os.path.basename(img_full_path)
        img_file_name = img_file_path.split(args.split_obj_name)[0]  # {img_file_name}_Sphere_x_background.png
        if img_file_name in processed_set:
            continue
        # Command
        output_path = os.path.join(args.output_folder, img_file_name)
        os.makedirs(output_path, exist_ok=True)
        cmd = f"python {args.script_path} --device {args.device} --background '{img_full_path}'"
        cmd += f" --output '{output_path}'"
        cmd += f" --stride {args.stride}"
        os.system(cmd)
        processed_set.add(img_file_name)
        print(f"Processed image {i + 1}/{len(img_list)}: {img_file_name}.")
    print(f"\nDone: processed {len(processed_set)} images.")
