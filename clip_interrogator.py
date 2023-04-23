import argparse
import os
from glob import glob

from models.interrogator import ClipInterrogator

def get_image_paths(image_dir):
    image_exts = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []

    for ext in image_exts:
        image_paths.extend(glob(os.path.join(image_dir, ext), recursive=True))

    return image_paths

def save_text_to_file(text, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', default='examples')
    parser.add_argument('--out_image_dir', default='output_desc')

    parser.add_argument('--interrogator_blip_model', choices=['blip-base','blip-large', 'blip2-2.7b', 'git-large-coco'], dest='interrogator_blip_model', default='blip-large', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--interrogator_clip_model', choices=['ViT-L-14/openai', 'ViT-H-14/laion2b_s32b_b79k'], dest='interrogator_clip_model', default='ViT-L-14/openai', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')

    args = parser.parse_args()

    clip_interrogator_model = ClipInterrogator(device="cuda",
                                                    clip_model=args.interrogator_clip_model,
                                                    blip_model=args.interrogator_blip_model)

    image_paths = get_image_paths(args.image_dir)

    for image_path in image_paths:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        out_text_path = os.path.join(args.out_image_dir, f'{file_name}.caption')

        caption = clip_interrogator_model.image_caption(image_path)

        save_text_to_file(caption, out_text_path)
