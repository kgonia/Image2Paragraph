import argparse
import os
from glob import glob

from models.image_text_transformation import ImageTextTransformation
from utils.util import display_images_and_text

def get_image_paths(image_dir):
    image_exts = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []

    for ext in image_exts:
        image_paths.extend(glob(os.path.join(image_dir, ext), recursive=True))

    return image_paths

def save_text_to_file(text, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(text)

def process_image(processor, image_src, out_image_path, out_text_path, out_negative_text_path):
    generated_text = processor.image_to_text(image_src)
    generated_negative_text = processor.image_to_negative_caption(image_src)
    generated_image = processor.text_to_image(generated_text)

    print(f"Processing: {image_src}")
    print("*" * 50)
    print("Generated Text:")
    print(generated_text)
    print("*" * 50)

    results = display_images_and_text(image_src, generated_image, generated_text, out_image_path)
    save_text_to_file(generated_text, out_text_path)
    save_text_to_file(generated_negative_text, out_negative_text_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_src', default='examples/1.jpg')
    parser.add_argument('--out_image_name', default='output/1_result.jpg')
    parser.add_argument('--image_dir', default='examples')
    parser.add_argument('--out_image_dir', default='output_desc')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
    parser.add_argument('--image_interrogator', action='store_true', dest='image_interrogator', default=True, help='Set this flag to True if you want to use Image Interrogator')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=True, help='Set this flag to True if you want to use semantic segmentation')
    parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_b', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
    parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
    parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, <6G GPU is not recommended>')
    parser.add_argument('--interrogator_blip_model', choices=['blip-base','blip-large', 'blip2-2.7b', 'git-large-coco'], dest='interrogator_blip_model', default='blip-large', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--interrogator_clip_model', choices=['ViT-L-14/openai', 'ViT-H-14/laion2b_s32b_b79k'], dest='interrogator_clip_model', default='ViT-L-14/openai', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')

    args = parser.parse_args()

    processor = ImageTextTransformation(args)

    if args.image_dir:
        # Process all images in the directory
        image_paths = get_image_paths(args.image_dir)

        for image_path in image_paths:
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            out_image_name = os.path.join(args.out_image_dir, f'{file_name}_result.jpg')
            out_text_path = os.path.join(args.out_image_dir, f'{file_name}.caption')
            out_negative_text_path = os.path.join(args.out_image_dir, f'{file_name}_negative.caption')

            process_image(processor, image_path, out_image_name, out_text_path, out_negative_text_path)

    else:
        # Process a single image
        file_name = os.path.splitext(os.path.basename(args.image_src))[0]
        out_text_path = os.path.join(args.out_image_dir, f'{file_name}.caption')
        out_negative_text_path = os.path.join(args.out_image_dir, f'{file_name}_negative.caption')

        process_image(processor, args.image_src, args.out_image_name, out_text_path, out_negative_text_path)

