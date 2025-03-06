import os
import glob
import argparse

import numpy as np

import image_util


def parse_argument():
    parser = argparse.ArgumentParser(description='Bash script for running inference on images using the Shadow Harmonization Network (pbla)')
    parser.add_argument("--input_render_folder", type=str, default="./test_data/collect_ours_userstudy")
    parser.add_argument("--input_net_out_folder", type=str, default='./out/collect_ours_userstudy')
    parser.add_argument("--image_filter", type=str, default="_background.png")
    parser.add_argument("--split_obj_name", type=str, default="_Sphere")
    parser.add_argument("--output_folder", type=str, default="./out/composite/")
    parser.add_argument("--device", type=str,
                        default='cuda:0')
    parser.add_argument("--vis", default=False, action="store_true")
    args = parser.parse_args()
    return args


def refine_direct(bg_img, gain_map, detection, mask):
    rgb_bg = (mask * detection) * bg_img
    rgb_fg = (((mask * (1 - detection)) * gain_map) * bg_img**2.2)**(1/2.2)

    # Gross per-channel code to enable vectorization without numerical issues
    # Red
    if (rgb_fg[:, :, 0][rgb_fg[:, :, 0] > 1e-1].shape[0] != 0) and (rgb_bg[:, :, 0][rgb_bg[:, :, 0] > 1e-1].shape[0] != 0):
        mean_bg_r = rgb_bg[:, :, 0][rgb_bg[:, :, 0] > 1e-1].mean()
        mean_fg_r = rgb_fg[:, :, 0][rgb_fg[:, :, 0] > 1e-1].mean()
        r_factor = (mean_bg_r / (mean_fg_r + np.finfo(float).eps)) ** 2.2
        gain_map[:, :, 0] = (gain_map * r_factor * (1 - detection) + gain_map * detection)[:, :, 0]

    # Green
    if (rgb_fg[:, :, 1][rgb_fg[:, :, 1] > 1e-1].shape[0] != 0) and (rgb_bg[:, :, 1][rgb_bg[:, :, 1] > 1e-1].shape[0] != 0):
        mean_bg_g = rgb_bg[:, :, 1][rgb_bg[:, :, 1] > 1e-1].mean()
        mean_fg_g = rgb_fg[:, :, 1][rgb_fg[:, :, 1] > 1e-1].mean()
        g_factor = (mean_bg_g / (mean_fg_g + np.finfo(float).eps)) ** 2.2
        gain_map[:, :, 1] = (gain_map * g_factor * (1 - detection) + gain_map * detection)[:, :, 1]

    # Blue
    if (rgb_fg[:, :, 2][rgb_fg[:, :, 2] > 1e-1].shape[0] != 0) and (rgb_bg[:, :, 2][rgb_bg[:, :, 2] > 1e-1].shape[0] != 0):
        mean_bg_b = rgb_bg[:, :, 2][rgb_bg[:, :, 2] > 1e-1].mean()
        mean_fg_b = rgb_fg[:, :, 2][rgb_fg[:, :, 2] > 1e-1].mean()
        b_factor = (mean_bg_b / (mean_fg_b + np.finfo(float).eps)) ** 2.2
        gain_map[:, :, 2] = (gain_map * b_factor * (1 - detection) + gain_map * detection)[:, :, 2]

    return gain_map


def shadow_harmonize(image_dict, visualize=False):
    # Images
    bg_linear = image_dict["background"]
    bg = image_util.rgb_to_srgb(bg_linear)
    obj_shadow = image_dict["shadow_mask"]
    obj_mask = image_dict["object_mask"]
    highlights = np.ones_like(image_dict["background"])  # no env shadows
    fg_comp = image_dict["foreground"]
    gain_ori = image_dict["gain"]
    det = image_dict["det"]

    # Refining the network output
    gain = refine_direct(bg, gain_ori.copy(), det, obj_shadow)
    # print(f"diff: {np.abs(gain_ori - gain).max()}")

    # Compositing the background portion
    bg_comp = (obj_shadow * bg_linear * gain + (1 - obj_shadow) * bg_linear) * highlights

    # Compositing background and foreground together and tone-mapping
    comp = obj_mask * fg_comp + (1 - obj_mask) * bg_comp
    comp_srgb = image_util.rgb_to_srgb(comp)

    # Visualization
    if visualize:

        vis = [bg, obj_shadow, gain_ori, gain, det,
               highlights, obj_mask, fg_comp, bg_comp, comp_srgb]
        titles = ["bg", "obj_shadow", "gain_ori", "gain", "det",
                  "highlights", "obj_mask", "fg_comp", "bg_comp", "comp_srgb"]
        image_util.plot_images(vis, titles, columns=5, figsize_base=8, show=True)
    return comp_srgb, 1.0 - det.clip(min=0.0, max=1.0)


if __name__ == "__main__":
    # Parse arguments
    args = parse_argument()
    assert os.path.exists(args.input_render_folder), f"Input render folder: {args.input_render_folder} does not exist"
    assert os.path.exists(args.input_net_out_folder), f"Input network output folder: {args.input_net_out_folder} does not exist"
    os.makedirs(args.output_folder, exist_ok=True)

    # Get the list of images
    img_list = glob.glob(os.path.join(args.input_render_folder, "*" + args.image_filter))
    img_list = sorted(img_list)
    print(f"Found {len(img_list)} images in {args.input_render_folder} with filter {args.image_filter}.")

    # Run compositing
    for i in range(0, len(img_list)):
        # Image file paths
        img_full_path = img_list[i]
        img_file_path = os.path.basename(img_full_path)
        img_file_name = img_file_path.split(args.split_obj_name)[0]  # {img_file_name}_Sphere_x_background.png
        obj_name = img_file_path[len(img_file_name) + 1:-len(args.image_filter)]
        print(f"Processing image {i + 1}/{len(img_list)}: {img_file_name} with object {obj_name}.")

        # Image dict
        image_path_dict = {
            "background": img_full_path,
            "foreground": img_full_path.replace("background.png", "foreground.exr"),
            "shadow_mask": img_full_path.replace("background.png", "shadow_mask.exr"),
            "object_mask": img_full_path.replace("background.png", "object_mask.exr"),
        }
        image_dict = {}
        for k, v in image_path_dict.items():
            assert os.path.exists(v), f"Image {k} does not exist: {v}"
            img = image_util.read_image(v, "numpy", inf_v=1000.0, nan_v=0.0, preserve_alpha=True)[:, :, :3]  # H X W X 3
            img = img.clip(min=0.0)
            if k in ["background"]:
                img = image_util.srgb_to_rgb(img)  # convert to linear space
            if k in ["shadow_mask", "object_mask"]:
                img = img.clip(min=0.0, max=1.0)
                img[img > 0.999] = 1.0  # Blender does not output perfect mask
            image_dict[k] = img

        # Network output
        shadow_net_output_path = os.path.join(args.input_net_out_folder, img_file_name, "network_output.npz")
        assert os.path.exists(shadow_net_output_path), f"Network output does not exist: {shadow_net_output_path}"
        shadow_net_out = np.load(shadow_net_output_path)
        image_dict["gain"] = shadow_net_out["gain"]
        image_dict["det"] = shadow_net_out["det"]
        composite, shadow_detect = shadow_harmonize(image_dict, visualize=args.vis)
        out_composite_path = os.path.join(args.output_folder, img_file_path.replace("background.png", "composite.png"))
        out_shadow_detect_path = os.path.join(args.output_folder, img_file_path.replace("background.png", "shadow_detect.png"))
        image_util.save_image_opencv(composite, out_composite_path, convert_2_uint8=True, is_RGB_img=True)
        image_util.save_image_opencv(shadow_detect, out_shadow_detect_path, convert_2_uint8=True, is_RGB_img=True)
    print("Done.")

