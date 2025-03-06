import os
import glob
import argparse
import shutil


def parse_argument():
    parser = argparse.ArgumentParser(description='Bash script for running inference on images using the Shadow Harmonization Network (pbla)')
    parser.add_argument("--input_ours_folder", type=str, default="./test_data/collect_ours_userstudy")
    parser.add_argument("--input_shadowcomp_folder", type=str, default='./out/composite')
    parser.add_argument("--image_filter", type=str, default="_background.png")
    parser.add_argument("--output_folder", type=str, default="./out/static_images/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_argument()
    assert os.path.exists(args.input_ours_folder), f"Input folder: {args.input_ours_folder} does not exist"
    assert os.path.exists(args.input_shadowcomp_folder), f"Input folder: {args.input_shadowcomp_folder} does not exist"

    # Get the list of images
    img_list = glob.glob(os.path.join(args.input_ours_folder, "*" + args.image_filter))
    img_list = sorted(img_list)
    print(f"Found {len(img_list)} images in {args.input_ours_folder} with filter {args.image_filter}.")

    # Output folders
    output_ours_folder = os.path.join(args.output_folder, "ours")
    output_shadowcomp_folder = os.path.join(args.output_folder, "shadow_harmonize")
    os.makedirs(output_ours_folder, exist_ok=True)
    os.makedirs(output_shadowcomp_folder, exist_ok=True)

    # Reorganize images
    img_file_name_list = []
    for i in range(0, len(img_list)):
        # Image file paths
        img_full_path = img_list[i]
        img_file_path = os.path.basename(img_full_path)  # {img_file_name}_Sphere_x_background.png
        img_file_name = img_file_path.split("_Sphere")[0]
        obj_name = img_file_path[len(img_file_name) + 1:-len("_background.png")]
        img_file_name_list.append(img_file_path[:-len("_background.png")])
        print(f"Processing image {i + 1}/{len(img_list)}: {img_file_name} with object {obj_name}.")

        # Image dict
        image_path_dict = {
            "background": img_full_path,
            "ours": img_full_path.replace("background.png", "ours.png"),
            "ours_wo_refine": img_full_path.replace("background.png", "ours_wo_refine.png"),
            "shadow_harmonize": img_full_path.replace(args.input_ours_folder, args.input_shadowcomp_folder).replace("background.png", "composite.png"),
        }

        # Copy images
        for k in image_path_dict.keys():
            assert os.path.exists(image_path_dict[k]), f"Image {k} does not exist: {image_path_dict[k]}"
            src_path = image_path_dict[k]
            if k in ["shadow_harmonize"]:
                dst_path = src_path.replace(args.input_shadowcomp_folder, output_shadowcomp_folder)
                dst_path = dst_path.replace("composite", "sh")
            elif k in ["background", "ours", "ours_wo_refine"]:
                dst_path = src_path.replace(args.input_ours_folder, output_ours_folder)
            else:
                assert False, f"Unknown key: {k}"
            if src_path.endswith(".png"):
                dst_path = dst_path.replace(".png", ".jpg")
                cmd = "convert"
                cmd += f" '{src_path}'"
                cmd += f" -quality 99%"
                cmd += f" '{dst_path}'"
                os.system(cmd)
            elif src_path.endswith(".jpg") or src_path.endswith(".jpeg"):
                dst_path.replace(".jpeg", ".jpg")
                shutil.copy(src_path, dst_path)
            else:
                assert False, f"Unknown image format: {src_path}"

    # Save list file
    with open(os.path.join(args.output_folder, "img_list.txt"), "w") as f:
        for img_file_name in img_file_name_list:
            f.write(f"{img_file_name}\n")
    print("Done.")

